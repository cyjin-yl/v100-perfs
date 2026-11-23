#!/usr/bin/env bash
# Bootstrap for the replacement owned Crater container after missing sshd.
# No local execution beyond bash -n has been performed.
# Whole bootstrap, including apt, is bounded to 4 hours. To clip the container
# to the original queue deadline as well, supply MXFP4_DEADLINE_UTC explicitly.
# This script does not submit jobs, alter a source-ready publication, or run CUDA.
set -Eeuo pipefail
export PATH="${PATH:?image_PATH_missing}:/usr/local/sbin:/usr/sbin:/sbin"

bootstrap_ttl=14400
bootstrap_now=$(date -u +%s)
if [[ -n ${MXFP4_DEADLINE_UTC:-} ]]; then
    bootstrap_deadline=$(date -u -d "$MXFP4_DEADLINE_UTC" +%s)
    bootstrap_remaining=$((bootstrap_deadline - bootstrap_now))
    if (( bootstrap_remaining < bootstrap_ttl )); then bootstrap_ttl=$bootstrap_remaining; fi
fi
(( bootstrap_ttl > 30 )) || { printf '%s\n' 'MXFP4_BOOTSTRAP_ERROR=deadline_expired' >&2; exit 70; }
command -v timeout >/dev/null
# TERM is sent 15 seconds before the bound; KILL ends the remaining process
# group at the bound. Pod/container termination also ends SSH child sessions.
exec timeout --signal=TERM --kill-after=15s "$((bootstrap_ttl - 15))s" /bin/bash -s <<'MXFP4_BOOTSTRAP_BODY'
set -Eeuo pipefail
umask 077
fail() { printf 'MXFP4_BOOTSTRAP_ERROR=%s\n' "$*" >&2; exit 70; }
[[ $(id -u) == 0 ]] || fail 'container_must_start_as_root'
for tool in getent stat install mktemp useradd usermod groupadd runuser bash python3 timeout sha256sum tee; do
    command -v "$tool" >/dev/null || fail "required_tool_missing:$tool"
done
bootstrap_runtime=$(mktemp -d /run/mxfp4-bootstrap.XXXXXXXX)
chmod 700 "$bootstrap_runtime"
policy_active=0
policy_saved=0
restore_policy() {
    if (( policy_active )); then
        if (( policy_saved )); then
            # A signal may arrive just before/after mv. A missing backup means
            # either the original never moved or it has already been restored.
            if [[ -e "$bootstrap_runtime/policy-rc.d.original" || -L "$bootstrap_runtime/policy-rc.d.original" ]]; then
                mv -f -- "$bootstrap_runtime/policy-rc.d.original" /usr/sbin/policy-rc.d || return 1
            fi
        else
            rm -f -- /usr/sbin/policy-rc.d || return 1
        fi
        policy_active=0
    fi
}
cleanup() {
    bootstrap_status=$?
    trap - EXIT
    if ! restore_policy; then
        printf '%s\n' 'MXFP4_BOOTSTRAP_ERROR=policy_restore_failed' >&2
        bootstrap_status=70
    fi
    exit "$bootstrap_status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ ! -x /usr/sbin/sshd ]]; then
    command -v apt-get >/dev/null || fail 'sshd_missing_and_no_apt_get'
    # Preserve any existing policy, including a symlink. Suppress package
    # postinst service starts before a default port-22 daemon can be launched.
    if [[ -e /usr/sbin/policy-rc.d || -L /usr/sbin/policy-rc.d ]]; then policy_saved=1; fi
    policy_active=1
    if (( policy_saved )); then
        mv -- /usr/sbin/policy-rc.d "$bootstrap_runtime/policy-rc.d.original"
    fi
    printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d
    chmod 755 /usr/sbin/policy-rc.d
    printf '%s\n' 'MXFP4_BOOTSTRAP_STAGE=install_openssh_server'
    apt_options=(-o DPkg::Lock::Timeout=30 -o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20)
    timeout --signal=TERM --kill-after=10s 240s env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" update
    timeout --signal=TERM --kill-after=10s 240s env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y --no-install-recommends openssh-server
    restore_policy
fi
[[ -x /usr/sbin/sshd ]] || fail 'sshd_still_missing'
command -v ssh-keygen >/dev/null || fail 'ssh_keygen_missing'
[[ -r /etc/pam.d/sshd ]] || fail 'pam_sshd_configuration_missing'
[[ ! -e /run/nologin && ! -e /etc/nologin ]] || fail 'container_has_nologin_marker'

bootstrap_keys=/workspace/user/.ssh/authorized_keys
[[ -f $bootstrap_keys && -s $bootstrap_keys ]] || fail 'persistent_authorized_keys_missing_or_empty'
login_uid=$(stat -Lc %u "$bootstrap_keys")
login_gid=$(stat -Lc %g "$bootstrap_keys")
if [[ $login_uid == 0 ]]; then
    # Platform-provided NB_UID is preferable to an unrelated UID 1000 if the
    # authorized_keys file was copied as root. Never change shared file owners.
    if [[ ${NB_UID:-} =~ ^[1-9][0-9]*$ ]]; then login_uid=$NB_UID; else login_uid=1000; fi
fi
if [[ $login_gid == 0 && ${NB_GID:-} =~ ^[1-9][0-9]*$ ]]; then login_gid=$NB_GID; fi
[[ $login_uid =~ ^[1-9][0-9]*$ ]] || fail 'invalid_nonroot_login_uid'
login_record=$(getent passwd "$login_uid" || true)
if [[ -n $login_record ]]; then
    login=${login_record%%:*}
else
    login=mxfp4dev
    ! getent passwd "$login" >/dev/null || fail 'mxfp4dev_name_belongs_to_another_uid'
    if [[ $login_gid == 0 ]]; then login_gid=$login_uid; fi
    if ! getent group "$login_gid" >/dev/null; then
        bootstrap_group=mxfp4g${login_gid}
        ! getent group "$bootstrap_group" >/dev/null || fail 'group_name_collision'
        groupadd --gid "$login_gid" "$bootstrap_group"
    fi
    useradd --uid "$login_uid" --gid "$login_gid" --create-home --shell /bin/bash "$login"
fi
[[ $login =~ ^[a-z_][a-z0-9_-]{0,31}$ && $login != root ]] || fail 'invalid_login_name'
[[ $(id -u "$login") == "$login_uid" ]] || fail 'login_uid_mismatch'
usermod --home /workspace/user --shell /bin/bash "$login"
login_gid=$(id -g "$login")
runuser -u "$login" -- /bin/sh -c 'test -d /workspace/user && test -x /workspace/user && test -w /workspace/user' || fail 'login_cannot_write_persistent_workspace'

# Public-key reads happen after dropping to the target user. Both the directory
# and the file need correct ownership and modes, not just authorized_keys.
[[ ! -e /run/mxfp4-ssh && ! -L /run/mxfp4-ssh ]] || fail 'ssh_runtime_directory_already_exists'
install -d -m 700 -o "$login_uid" -g "$login_gid" /run/mxfp4-ssh
install -m 600 -o "$login_uid" -g "$login_gid" "$bootstrap_keys" /run/mxfp4-ssh/authorized_keys
runuser -u "$login" -- ssh-keygen -lf /run/mxfp4-ssh/authorized_keys > "$bootstrap_runtime/authorized-key-fingerprints"
install -d -m 755 -o 0 -g 0 /run/sshd
ssh-keygen -q -t ed25519 -N '' -f "$bootstrap_runtime/ssh_host_ed25519_key"

# Preserve the image's Python/CUDA environment in the SSH child session.
# Only path characters are accepted before interpolating literal sshd values.
[[ $PATH =~ ^[a-zA-Z0-9_./:+-]+$ ]] || fail 'unsupported_character_in_image_PATH'
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
    [[ $LD_LIBRARY_PATH =~ ^[a-zA-Z0-9_./:+-]+$ ]] || fail 'unsupported_character_in_image_LD_LIBRARY_PATH'
fi
runuser -u "$login" -- env PATH="$PATH" python3 -c 'import sys; print("MXFP4_IMAGE_PYTHON=" + sys.executable)'

cat > "$bootstrap_runtime/sshd_config" <<MXFP4_SSHD_CONFIG
Port 2222
AddressFamily inet
ListenAddress 0.0.0.0
HostKey $bootstrap_runtime/ssh_host_ed25519_key
PidFile $bootstrap_runtime/sshd.pid
AuthorizedKeysFile /run/mxfp4-ssh/authorized_keys
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
    printf 'SetEnv LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH" >> "$bootstrap_runtime/sshd_config"
fi
/usr/sbin/sshd -t -f "$bootstrap_runtime/sshd_config"
/usr/sbin/sshd -T -f "$bootstrap_runtime/sshd_config" > "$bootstrap_runtime/sshd-effective-config"

# These are the runner's exact authenticated-log markers. Do not print the
# client authorized_keys file: it would be parsed as a conflicting HOST key.
printf 'MXFP4_DEV_USER=%s\n' "$login"
cat "$bootstrap_runtime/ssh_host_ed25519_key.pub"
printf 'MXFP4_BOOTSTRAP_STAGE=sshd_foreground uid=%s gid=%s\n' "$login_uid" "$login_gid"
restore_policy
trap - EXIT TERM INT
exec /usr/sbin/sshd -D -e -f "$bootstrap_runtime/sshd_config"
MXFP4_BOOTSTRAP_BODY
