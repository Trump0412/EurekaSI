"""User-operated hidden Cookie login; credentials never become process arguments.

Linux only. Run directly in your private terminal, not an agent tool session.
The third-party client retains credentials in its private configuration.
"""
import argparse
import getpass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def normalize_cookie(raw):
    value = raw.strip()
    if value.lower().startswith('cookie:'):
        value = value[7:].strip()
    if not value or len(value) > 65536 or any(ord(c) < 32 or ord(c) > 126 for c in value):
        raise ValueError('Cookie must be a single ASCII request-header value')
    pairs = {}
    for part in value.split(';'):
        if not part.strip(): continue
        name, separator, token = part.strip().partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_!#$%&*+.^|~-]+', name):
            raise ValueError('Invalid Cookie field syntax')
        if any(c.isspace() or c in '\"\'\\' for c in token):
            raise ValueError('Quoted, escaped or whitespace-containing Cookie values are not accepted')
        if name in pairs: raise ValueError('Duplicate Cookie fields; copy one complete request header')
        pairs[name] = token
    if not pairs.get('BDUSS'): raise ValueError('BDUSS is missing; copy the Cookie header of your logged-in account')
    return '; '.join(f'{key}={value}' for key,value in pairs.items()), bool(pairs.get('STOKEN'))


def fingerprint(path):
    if not path.exists(): return None
    stat = path.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def execute(binary, config_dir, commands):
    env = dict(os.environ, BAIDUPCS_GO_CONFIG_DIR=str(config_dir), BAIDUPCS_GO_VERBOSE='0')
    # No shell, no cookie in argv/environment, no client output forwarded to logs.
    return subprocess.run([str(binary)], input=commands, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env, timeout=120, check=False)


def login(root, cookie=None, self_test=False):
    if os.name != 'posix': raise ValueError('Use this entry in the Linux server terminal')
    if not self_test: cookie, _ = normalize_cookie(cookie)
    root = Path(root).resolve(strict=True)
    binary = root/'tools/baidupcs/v4.0.2/BaiduPCS-Go'
    if not binary.is_file(): raise ValueError('Install the pinned client first')
    private = root/'.private'
    if private.is_symlink(): raise ValueError('Private directory must not be a symlink')
    private.mkdir(mode=0o700, exist_ok=True); private.chmod(0o700)
    target_dir = private/'baidu'
    if target_dir.is_symlink(): raise ValueError('Private config directory must not be a symlink')
    target_dir.mkdir(mode=0o700, exist_ok=True); target_dir.chmod(0o700)
    target = target_dir/'pcs_config.json'
    if target.is_symlink(): raise ValueError('Private config file must not be a symlink')
    import fcntl
    with (private/'baidu-cookie-login.lock').open('a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        before = fingerprint(target)
        with tempfile.TemporaryDirectory(prefix='baidu-login-', dir=private) as directory:
            temporary = Path(directory)
            # REPL appends every command to history. Sink it before startup;
            # never write a Cookie command into a regular history file.
            history = temporary/'pcs_command_history.txt'
            history.symlink_to('/dev/null')
            if not self_test and target.exists():
                shutil.copyfile(target, temporary/'pcs_config.json')
                os.chmod(temporary/'pcs_config.json', 0o600)
            commands = b'help\n' if self_test else ('login -cookies="'+cookie+'"\n').encode('ascii')
            result = execute(binary, temporary, commands)
            if not history.is_symlink() or os.readlink(history) != '/dev/null':
                raise RuntimeError('History sink changed; refusing to retain credentials')
            if self_test:
                if result.returncode or b'login' not in result.stdout:
                    raise RuntimeError('Client stdin/REPL compatibility check failed')
                return 'Self-test passed; no account authentication attempted.'
            if result.returncode or '百度帐号登录成功:'.encode() not in result.stdout:
                raise RuntimeError('Cookie login was not confirmed. Refresh the Cookie or check network/account verification; client output is hidden to protect secrets.')
            saved = temporary/'pcs_config.json'
            if not saved.is_file() or saved.is_symlink(): raise RuntimeError('Client did not save a regular account configuration')
            if fingerprint(target) != before: raise RuntimeError('Account configuration changed concurrently; retry when other client commands finish')
            saved.chmod(0o600); os.replace(saved,target)
    return 'Cookie login confirmed. Configuration saved privately; run baidu-cli.sh ls to verify file access.'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true',help='Test client stdin/history handling without login')
    args=p.parse_args(); os.umask(0o077)
    if os.name=='posix':
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root=os.environ.get('NODE_ROOT')
    if not root: raise ValueError('Export NODE_ROOT before running this entry')
    if args.self_test: print(login(root,self_test=True)); return
    if not sys.stdin.isatty() or not sys.stderr.isatty():
        raise ValueError('Private interactive terminal required; input redirection and agent sessions are refused')
    print('Third-party client. Paste only your own logged-in pan.baidu.com request Cookie; input is hidden.')
    cookie, has_stoken=normalize_cookie(getpass.getpass('Cookie (hidden): '))
    if not has_stoken: print('STOKEN is absent; share-transfer operations may be unavailable.')
    print(login(root,cookie))


if __name__=='__main__':
    try: main()
    except (Exception, KeyboardInterrupt):
        # Never print exception values: subprocess exceptions may contain stdin.
        print('Login/check did not complete. Check input format, BDUSS, network, and account verification. No raw client output or credentials are displayed.',file=sys.stderr)
        sys.exit(1)
