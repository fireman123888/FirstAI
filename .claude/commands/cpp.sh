#!/bin/bash

# 檢查是否有提供提交訊息
if [ -z "$1" ]; then
  echo "錯誤：請提供提交訊息！"
  echo "用法: ./cpp.sh \"您的提交訊息\""
  exit 1
fi

# 1. 將所有變更加入暫存區
git add .
echo "✅ 檔案已加入暫存區"

# 2. 提交變更 (使用第一個參數作為訊息)
git commit -m "$1"
echo "✅ 檔案已提交"

# 3. 推送到遠端倉庫
#   (自動獲取當前分支名稱)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$CURRENT_BRANCH"
echo "✅ 變更已推送到 origin/$CURRENT_BRANCH"

# 4. 使用 GitHub CLI 建立 Pull Request
#   --fill 會自動使用您的 commit 訊息作為標題和內文
gh pr create --fill
echo "🎉 Pull Request 已成功建立！"
