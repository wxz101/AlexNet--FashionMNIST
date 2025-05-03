#!/usr/bin/env python3
# 文件名：github_commit.py
# 适用于PyCharm终端（自动识别系统类型）

import os
import sys
import platform
import subprocess


def run_command(cmd, shell_type=None):
    """执行命令并处理错误"""
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                executable=shell_type)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误：{e.stderr.decode()}")
        sys.exit(1)


def main():
    # 配置区（用户需修改部分）
    repo_url = "https://github.com/你的用户名/你的仓库名.git"
    default_branch = "main"  # 或 "master"

    # 自动检测系统类型
    is_windows = platform.system() == "Windows"
    shell_type = "cmd" if is_windows else "bash"

    # 获取提交信息
    commit_msg = input("请输入提交说明（或直接回车使用默认信息）: ") or "PyCharm自动提交"

    # 1. 检查Git安装
    print("🔍 检查Git环境...")
    run_command("git --version", shell_type)

    # 2. 初始化仓库（如需要）
    if not os.path.exists(".git"):
        print("🆕 初始化Git仓库...")
        run_command("git init", shell_type)

    # 3. 设置远程仓库
    remotes = run_command("git remote", shell_type)
    if "origin" not in remotes.split():
        print("🌐 添加远程仓库...")
        run_command(f"git remote add origin {repo_url}", shell_type)
    else:
        print("🔄 更新远程仓库URL...")
        run_command(f"git remote set-url origin {repo_url}", shell_type)

    # 4. 检查文件状态
    print("📊 检查文件变更...")
    status = run_command("git status -s", shell_type)
    if not status:
        print("⚠️ 没有检测到文件变更，跳过提交")
        sys.exit(0)

    # 5. 执行提交
    print("💾 正在提交文件...")
    run_command("git add .", shell_type)
    run_command(f'git commit -m "{commit_msg}"', shell_type)

    # 6. 推送代码
    print("🚀 推送至GitHub...")
    current_branch = run_command("git branch --show-current", shell_type) or default_branch
    run_command(f"git push -u origin {current_branch}", shell_type)

    print(f"✅ 成功提交到 {current_branch} 分支！")


if __name__ == "__main__":
    main()