#!/usr/bin/env bash
# One-shot preparation for the already authorized, running own Crater pod.
# Does not start sshd or change shared workspace/key ownership or permissions.
# PID1's existing bootstrap timeout remains the lifetime/deadline authority.
set -Eeuo pipefail
umask 077
export PATH="${PATH:?image_PATH_missing}:/usr/local/sbin:/usr/sbin:/sbin"
fail() { printf 'MXFP4_REPAIR_ERROR=%s\n' "$*" >&2; exit 70; }
[[ $(id -u) == 0 ]] || fail 'repair_requires_own_container_root'
for tool in stat getent install mkdir cp useradd usermod groupadd runuser ssh-keygen; do
    command -v "$tool" >/dev/null || fail "required_tool_missing:$tool"
done
[[ -x /usr/sbin/sshd && -r /etc/pam.d/sshd ]] || fail 'installed_sshd_or_pam_missing'
[[ ! -e /run/nologin && ! -e /etc/nologin ]] || fail 'container_has_nologin_marker'
[[ $PATH =~ ^[a-zA-Z0-9_./:+-]+$ ]] || fail 'unsupported_character_in_image_PATH'
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
    [[ $LD_LIBRARY_PATH =~ ^[a-zA-Z0-9_./:+-]+$ ]] || fail 'unsupported_character_in_image_LD_LIBRARY_PATH'
fi

workspace=/workspace/user
source_keys=$workspace/.ssh/authorized_keys
[[ -d $workspace ]] || fail 'own_workspace_missing'
[[ $(stat -Lc '%u:%g:%a' "$workspace") == '1001:1001:750' ]] || fail 'workspace_identity_or_mode_differs_from_observed_1001_1001_750'
[[ -f $source_keys && -s $source_keys && -r $source_keys ]] || fail 'existing_authorized_keys_unreadable'
shared_before=$(stat -Lc '%u:%g:%a' "$workspace" "$source_keys")
printf 'MXFP4_REPAIR_WORKSPACE uid=%s gid=%s mode=%s\n' "$(stat -Lc %u "$workspace")" "$(stat -Lc %g "$workspace")" "$(stat -Lc %a "$workspace")"
printf 'MXFP4_REPAIR_KEYS uid=%s gid=%s mode=%s\n' "$(stat -Lc %u "$source_keys")" "$(stat -Lc %g "$source_keys")" "$(stat -Lc %a "$source_keys")"

# mkdir is the one-shot lock. It is deliberately retained even after a failed
# attempt; a second invocation must stop for explicit operator inspection.
repair_runtime=/run/mxfp4-owner-workspace-ssh.once
mkdir -m 700 -- "$repair_runtime" || fail 'repair_already_attempted_or_runtime_unavailable'
keys_runtime=/run/mxfp4-owner-workspace-keys
[[ ! -e $keys_runtime && ! -L $keys_runtime ]] || fail 'private_keys_runtime_already_exists'
[[ ! -L /etc/ssh/sshd_config ]] || fail 'sshd_config_must_not_be_a_symlink'
if [[ -e /etc/ssh/sshd_config ]]; then
    [[ -f /etc/ssh/sshd_config ]] || fail 'sshd_config_is_not_a_regular_file'
    cp -p -- /etc/ssh/sshd_config "$repair_runtime/sshd_config.original"
fi

login_uid=1001
login_record=$(getent passwd "$login_uid" || true)
if [[ -n $login_record ]]; then
    login=${login_record%%:*}
else
    login=mxfp4owner
    ! getent passwd "$login" >/dev/null || fail 'repair_login_name_belongs_to_another_uid'
    if ! getent group 1001 >/dev/null; then
        ! getent group mxfp4owner1001 >/dev/null || fail 'repair_group_name_collision'
        groupadd --gid 1001 mxfp4owner1001
    fi
    useradd --uid 1001 --gid 1001 --no-create-home --home-dir "$workspace" --shell /bin/bash "$login"
fi
[[ $login =~ ^[a-z_][a-z0-9_-]{0,31}$ && $login != root ]] || fail 'invalid_nonroot_login_name'
[[ $(id -u "$login") == 1001 ]] || fail 'login_uid_is_not_workspace_owner'
# Reuse an existing account's primary group. In particular, do not run
# usermod --gid, which could change group ownership within its shared home.
usermod --home "$workspace" --shell /bin/bash "$login"
login_gid=$(id -g "$login")
runuser -u "$login" -- /bin/sh -c 'test -d /workspace/user && test -x /workspace/user && test -w /workspace/user' || fail 'workspace_owner_cannot_write_own_workspace'

# Directly under /run: the nonroot login can traverse this 0700 directory;
# it cannot traverse the separate root-only host-key/config backup directory.
install -d -m 700 -o 1001 -g "$login_gid" "$keys_runtime"
install -m 600 -o 1001 -g "$login_gid" "$source_keys" "$keys_runtime/authorized_keys"
runuser -u "$login" -- ssh-keygen -lf "$keys_runtime/authorized_keys" > "$repair_runtime/client-key-fingerprints"
[[ ! -L /run/sshd ]] || fail 'sshd_runtime_must_not_be_a_symlink'
install -d -m 755 -o 0 -g 0 /run/sshd
ssh-keygen -q -t ed25519 -N '' -f "$repair_runtime/ssh_host_ed25519_key"

cat > "$repair_runtime/sshd_config" <<MXFP4_SSHD_CONFIG
Port 22
AddressFamily inet
ListenAddress 0.0.0.0
HostKey $repair_runtime/ssh_host_ed25519_key
PidFile /run/sshd.pid
AuthorizedKeysFile $keys_runtime/authorized_keys
AllowUsers $login
AuthenticationMethods publickey
PubkeyAuthentication yes
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
UsePAM yes
StrictModes yes
PermitUserEnvironment no
HostbasedAuthentication no
GSSAPIAuthentication no
AllowAgentForwarding no
AllowTcpForwarding no
X11Forwarding no
PermitTunnel no
PermitTTY no
UseDNS no
LoginGraceTime 30
LogLevel VERBOSE
Subsystem sftp internal-sftp
SetEnv PATH=$PATH
MXFP4_SSHD_CONFIG
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
    printf 'SetEnv LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH" >> "$repair_runtime/sshd_config"
fi
/usr/sbin/sshd -t -f "$repair_runtime/sshd_config"
/usr/sbin/sshd -T -f "$repair_runtime/sshd_config" > "$repair_runtime/sshd-effective-config"
install -m 600 -o 0 -g 0 "$repair_runtime/sshd_config" /etc/ssh/sshd_config
/usr/sbin/sshd -t -f /etc/ssh/sshd_config
/usr/sbin/sshd -T -f /etc/ssh/sshd_config > "$repair_runtime/installed-effective-config"
[[ $(stat -Lc '%u:%g:%a' "$workspace" "$source_keys") == "$shared_before" ]] || fail 'shared_identity_or_modes_changed_during_repair'

# The frozen runner reads authenticated job logs, not terminal stdout. Only
# its exact user marker and one generated HOST public key go to PID1 stdout.
# No client authorized_keys contents are published, and no service is started.
host_public_key=$(< "$repair_runtime/ssh_host_ed25519_key.pub")
[[ $host_public_key == ssh-ed25519\ * && $host_public_key != *$'\n'* ]] || fail 'unexpected_host_public_key_format'
printf 'MXFP4_DEV_USER=%s\n%s\n' "$login" "$host_public_key" > "$repair_runtime/published-identity"
cat "$repair_runtime/published-identity" > /proc/1/fd/1
printf '%s\n' configured > "$repair_runtime/status"
printf 'MXFP4_REPAIR_CONFIGURED=1 uid=%s gid=%s service_started=0\n' "$login_uid" "$login_gid"
