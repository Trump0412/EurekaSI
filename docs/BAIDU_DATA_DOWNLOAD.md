# 百度网盘数据下载：手动认证，工具与训练隔离

此入口使用**第三方** `qjfoidnh/BaiduPCS-Go`，不是百度官方客户端。
2026-09-20 核验的最新稳定版为 [v4.0.2](https://github.com/qjfoidnh/BaiduPCS-Go/releases/tag/v4.0.2)。安装固定版本，直接从项目 GitHub Release 下载；不修改训练环境、不使用 GPU。

## 安装与权限

在自己的 SSH 终端中将 `NODE_ROOT` 设为独立的节点工作目录，然后进入 EurekaSI 仓库：

```bash
export NODE_ROOT=/absolute/path/to/node-root
bash scripts/setup-baidu-cli.sh "$NODE_ROOT"
bash scripts/baidu-cli.sh --version
bash scripts/baidu-cli.sh login --help
```

二进制保存在 `$NODE_ROOT/tools/baidupcs/v4.0.2/`，账号状态保存在 `$NODE_ROOT/.private/baidu/`，目录权限 0700，启动使用 umask 077。配置目录由上游支持的 `BAIDUPCS_GO_CONFIG_DIR` 指定；见[上游实现](https://github.com/qjfoidnh/BaiduPCS-Go/blob/v4.0.2/main.go)。同一 Unix UID 或 root 仍能访问这些文件，所以不要把凭据存进多人共享账号环境。

安装器只检查版本，**不会登录、读取凭据或下载账号内文件**。安装成功不代表账号认证成功。

## 用户自行登录：重要限制

### 推荐入口：隐藏输入 Cookie

用户授权使用Cookie认证时，在自己的、未录屏且未录制输入的交互终端执行：

```bash
export NODE_ROOT=/absolute/path/to/node-root
cd "$NODE_ROOT/EurekaSI"
python3 scripts/baidu-cookie-login.py
```

在个人浏览器登录自己的 `pan.baidu.com`，打开开发者工具 Network，刷新页面，选择发往该站点的请求，复制 **Request Headers 的 Cookie 值**，不是响应的 Set-Cookie、不是整段“Copy as cURL”。在上面隐藏输入提示中粘贴整行并回车。需包含BDUSS；STOKEN缺失会提示转存功能可能不可用。不要将Cookie发给助手或放进命令参数、环境变量、截图、Git。

脚本把Cookie仅通过匿名stdin管道送入固定第三方客户端，不使用shell；客户端原始输出不转发、不落日志，独立临时配置中的命令历史预先指向 `/dev/null`。核实客户端成功标记后才原子更新私有配置（文件0600、目录0700），失败保留原账号配置。Cookie仍会按客户端功能需要保存在私有配置中，不能防范root、同UID进程或终端输入录制。此入口与训练隔离，不占用GPU。

可运行 `python3 scripts/baidu-cookie-login.py --self-test` 验证stdin与历史屏蔽，不进行认证。当前服务器自检及8项CPU测试通过，**不代表真实Cookie登录已成功**。成功提示后运行 `bash scripts/baidu-cli.sh ls` 检查网盘访问；认证成功也不保证转存/下载权限。

上游说明普通交互式 `login` 已长期不维护，不推荐依赖它；推荐的 Cookie 方式涉及敏感凭据。未验证二维码登录支持，不能承诺可登录。见[上游登录说明](https://github.com/qjfoidnh/BaiduPCS-Go#登录百度帐号)。

如愿意尝试，只在自己的、未被录屏或录制日志的 SSH 终端运行：

```bash
bash scripts/baidu-cli.sh login
```

不要将 Cookie、BDUSS、STOKEN、密码发给助手、写进 shell 参数或提交 Git。认证由用户在自己的终端完成。若登录失败，请停止反复重试，不要绕过平台验证；可在可信个人电脑使用[百度官方网盘](https://pan.baidu.com/download)下载，再将数据上传到服务器。官方 Linux 图形客户端的无显示服务器登录未验证，此处没有部署 GUI，也不能将 `ssh -X` 视为已经可用。

## 登录后由用户下载

先从数据集官方页面获取文件并转存到自己的网盘；遵循数据集许可。只下载已经获授权的文件。

```bash
bash scripts/baidu-cli.sh ls
bash scripts/baidu-cli.sh config set -savedir "$NODE_ROOT/downloads/baidu" -max_parallel 1 -max_download_load 1
bash scripts/baidu-cli.sh download '/网盘内数据目录'
```

默认先单连接、单文件，避免下载挤占训练共享存储。第三方工具不提供超出官方权限的下载提速；普通账号过高并发可能触发限速，见[上游配置说明](https://github.com/qjfoidnh/BaiduPCS-Go#显示和修改程序配置项)。成功下载后另行核对文件数量、解压结构、官方数据清单，再接入训练 manifest；不能只凭下载命令退出码宣称数据完整。
