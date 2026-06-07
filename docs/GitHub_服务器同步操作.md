# GitHub 到服务器同步操作

本文档说明以后如何把本地代码同步到 GitHub，再从服务器拉到训练环境。当前内容按已经验证成功的方案整理，适用于本机 Windows + 远端 Ubuntu 服务器。

## 适用范围

- 当前最推荐用于 `meta-model-train`
- 服务器侧推荐用 `git@github.com:<github-user>/<repo>.git`
- 当前服务器不建议走 GitHub HTTPS；已经验证更稳的是 `SSH over 443`

## 一次性配置

### 1. 本地准备 GitHub 仓库

建议把后续真正需要训练的代码单独放进一个仓库，避免把主仓库里无关改动一起同步上去。

如果本地目录已经是 Git 仓库，只需要补远端：

```powershell
git remote add origin https://github.com/<github-user>/<repo>.git
git branch -M main
git push -u origin main
```

如果远端已经存在 `origin`，先查看：

```powershell
git remote -v
```

### 2. 服务器准备 deploy key

在服务器生成一把专门给 GitHub 用的无密码 SSH key：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "server-<repo>" -f ~/.ssh/github_<repo>
```

说明：

- `Enter passphrase` 直接回车，留空
- 这样后续 `git pull` 不会反复要求输入密码

查看公钥：

```bash
cat ~/.ssh/github_<repo>.pub
```

把整行公钥复制到 GitHub 仓库：

- 路径：`Settings -> Deploy keys -> Add deploy key`
- `Title` 可填 `server-<repo>`
- `Allow write access` 先不要勾，读权限就够

### 3. 服务器配置 GitHub 走 443 端口

创建 `~/.ssh/config`：

```bash
cat > ~/.ssh/config <<'EOF'
Host github.com
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/github_<repo>
  IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
```

测试认证：

```bash
ssh -T git@github.com
```

如果成功，通常会看到类似输出：

```text
Hi <github-user>/<repo>! You've successfully authenticated, but GitHub does not provide shell access.
```

### 4. 服务器首次拉代码

```bash
cd /root
git clone git@github.com:<github-user>/<repo>.git
cd <repo>
git log --oneline -1
```

## 日常同步流程

以后每次改完代码，建议按下面顺序操作。

### 1. 本地提交并推送

先看状态：

```powershell
git status
```

只提交你想同步的内容：

```powershell
git add meta-model-train
git commit -m "update meta model training"
git push
```

### 2. 服务器拉取更新

```bash
cd /root/<repo>
git pull
git log --oneline -1
```

## 训练前检查

每次在服务器开训前，建议先检查以下几项：

```bash
cd /root/<repo>
git status
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

建议 `git status` 保持干净，避免训练时混入手改内容。

## 常见问题

### 1. 服务器 `git ls-remote https://github.com/...` 很慢或失败

这台服务器已经出现过 GitHub HTTPS 卡住、TLS 中断的问题，因此不要把服务器侧同步方案建立在 HTTPS 上。推荐始终使用：

```text
git@github.com:<github-user>/<repo>.git
```

并通过 `~/.ssh/config` 强制走 `ssh.github.com:443`。

### 2. `ssh -T git@github.com` 提示输入 passphrase

说明你生成的是带密码的私钥。训练服务器不太适合长期使用这种方式，建议重新生成一把无密码 deploy key。

### 3. 本地能 push，服务器不能 pull

优先检查：

- deploy key 是否加到了正确仓库
- `~/.ssh/config` 的 `IdentityFile` 是否指向正确私钥
- 远端 URL 是否是 `git@github.com:...`

