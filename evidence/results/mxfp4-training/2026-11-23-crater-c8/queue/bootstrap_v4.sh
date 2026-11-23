#!/usr/bin/env bash
# Bootstrap v4: map the authorized platform/workspace identity without changing
# shared ownership or permissions. Nonzero setup failures hold for diagnosis
# under the same outer timeout; a blocked container is not SSH-ready.
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
bootstrap_runtime=''
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
    bootstrap_original_status=$?
    bootstrap_status=$bootstrap_original_status
    trap - EXIT
    if ! restore_policy; then
        printf '%s\n' 'MXFP4_BOOTSTRAP_ERROR=policy_restore_failed' >&2
        bootstrap_status=70
    fi
    # A timeout or operator signal must never become a new waiting period.
    if (( bootstrap_original_status == 130 || bootstrap_original_status == 143 )); then
        exit "$bootstrap_original_status"
    fi
    if (( bootstrap_status != 0 )); then
        printf 'MXFP4_BOOTSTRAP_BLOCKED=exit_%s\n' "$bootstrap_status" >&2
        # No SSH-ready marker and no CUDA work. The outer GNU timeout keeps
        # both this hold and any later diagnosis within the original bound.
        command -v sleep >/dev/null || exit "$bootstrap_status"
        while :; do sleep 10; done
    fi
    exit "$bootstrap_status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

[[ $(id -u) == 0 ]] || fail 'container_must_start_as_root'
for tool in getent stat install mktemp useradd usermod groupadd runuser bash python3 timeout sha256sum tee sleep; do
    command -v "$tool" >/dev/null || fail "required_tool_missing:$tool"
done
bootstrap_runtime=$(mktemp -d /run/mxfp4-bootstrap.XXXXXXXX)
chmod 700 "$bootstrap_runtime"

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

bootstrap_workspace=/workspace/user
bootstrap_keys=$bootstrap_workspace/.ssh/authorized_keys
[[ -d $bootstrap_workspace ]] || fail 'persistent_workspace_directory_missing'
[[ -f $bootstrap_keys && -s $bootstrap_keys ]] || fail 'persistent_authorized_keys_missing_or_empty'
workspace_uid=$(stat -Lc %u "$bootstrap_workspace")
workspace_gid=$(stat -Lc %g "$bootstrap_workspace")
workspace_mode=$(stat -Lc %a "$bootstrap_workspace")
keys_uid=$(stat -Lc %u "$bootstrap_keys")
keys_gid=$(stat -Lc %g "$bootstrap_keys")
keys_mode=$(stat -Lc %a "$bootstrap_keys")
printf 'MXFP4_WORKSPACE_IDS uid=%s gid=%s mode=%s\n' "$workspace_uid" "$workspace_gid" "$workspace_mode"
printf 'MXFP4_AUTHORIZED_KEYS_IDS uid=%s gid=%s mode=%s\n' "$keys_uid" "$keys_gid" "$keys_mode"
# The key file may have been copied by another UID. Its owner is diagnostic
# only and never selects the process identity. Platform IDs have priority;
# without NB_UID, only the mounted own-workspace directory owner is usable.
valid_nonroot_id() {
    [[ $1 =~ ^[1-9][0-9]{0,9}$ ]] && (( 10#$1 <= 4294967294 ))
}
if [[ -n ${NB_UID:-} ]]; then
    valid_nonroot_id "$NB_UID" || fail 'invalid_nonroot_platform_NB_UID'
    login_uid=$NB_UID
else
    valid_nonroot_id "$workspace_uid" || fail 'NB_UID_missing_and_workspace_owner_not_nonroot'
    login_uid=$workspace_uid
fi
login_record=$(getent passwd "$login_uid" || true)
existing_primary_gid=''
if [[ -n $login_record ]]; then
    IFS=: read -r login passwd_marker recorded_uid existing_primary_gid passwd_gecos passwd_home passwd_shell <<< "$login_record"
    [[ $recorded_uid == "$login_uid" ]] || fail 'existing_login_uid_mismatch'
else
    login=mxfp4dev
    ! getent passwd "$login" >/dev/null || fail 'mxfp4dev_name_belongs_to_another_uid'
fi
[[ $login =~ ^[a-z_][a-z0-9_-]{0,31}$ && $login != root ]] || fail 'invalid_login_name'
if [[ -n ${NB_GID:-} ]]; then
    valid_nonroot_id "$NB_GID" || fail 'invalid_nonroot_platform_NB_GID'
    login_gid=$NB_GID
elif valid_nonroot_id "$workspace_gid"; then
    login_gid=$workspace_gid
elif valid_nonroot_id "$existing_primary_gid"; then
    login_gid=$existing_primary_gid
else
    fail 'NB_GID_missing_and_no_known_nonroot_primary_gid'
fi
printf 'MXFP4_PLATFORM_IDS uid=%s gid=%s\n' "${NB_UID:-unset}" "${NB_GID:-unset}"
printf 'MXFP4_SELECTED_IDS uid=%s gid=%s\n' "$login_uid" "$login_gid"
if ! getent group "$login_gid" >/dev/null; then
    bootstrap_group=mxfp4g${login_gid}
    ! getent group "$bootstrap_group" >/dev/null || fail 'group_name_collision'
    groupadd --gid "$login_gid" "$bootstrap_group"
fi
if [[ -z $login_record ]]; then
    # -M prevents useradd from creating/chowning an existing shared home.
    useradd --uid "$login_uid" --gid "$login_gid" --no-create-home --home-dir "$bootstrap_workspace" --shell /bin/bash "$login"
elif [[ $existing_primary_gid != "$login_gid" ]]; then
    # usermod -g can recursively change the old primary group's files in the
    # current home. Point the local passwd entry at a new private empty home
    # first, WITHOUT -m, so changing GID cannot touch the shared workspace.
    bootstrap_account_home=$bootstrap_runtime/account-home
    install -d -m 700 -o "$login_uid" -g "$existing_primary_gid" "$bootstrap_account_home"
    usermod --home "$bootstrap_account_home" "$login"
    updated_login_record=$(getent passwd "$login")
    IFS=: read -r verified_login verified_password verified_uid verified_gid verified_gecos verified_home verified_shell <<< "$updated_login_record"
    [[ $verified_uid == "$login_uid" && $verified_home == "$bootstrap_account_home" ]] || fail 'private_account_home_not_applied'
    usermod --gid "$login_gid" "$login"
fi
# No --move-home, --uid, chown, chmod, or ACL update targets the workspace.
usermod --home "$bootstrap_workspace" --shell /bin/bash "$login"
[[ $(id -u "$login") == "$login_uid" ]] || fail 'login_uid_mismatch'
[[ $(id -g "$login") == "$login_gid" ]] || fail 'login_gid_mismatch'
printf 'MXFP4_LOGIN_IDS uid=%s gid=%s\n' "$(id -u "$login")" "$(id -g "$login")"
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
