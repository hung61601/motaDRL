![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge\&logo=windows\&logoColor=white)
[![coverage badge](./coverage.svg)](https://hung61601.github.io/motaDRL/tests/coverage/)  

# Mota DRL

Mota DRL 是「使用深度強化學習的魔塔 AI」(Mota AI using Deep Reinforcement Learning) 的簡寫。

本專案實現了使用 Graph Isomorphism Network (GIN) 結合 Proximal Policy Optimization (PPO) 來學習遊戲策略。

- 作者：Hung1

## 相依套件 Dependencies
- Python 版本為 3.11.5
- 安裝相依套件可在根目錄下使用指令：
```bash
pip install -r requirements.txt
```

## 操作介紹 Instruction
### 訓練模型
您可以執行在`run_scripts/`目錄下帶有`training`開頭的檔案，將會使用對應的演算法開始訓練模型。
訓練模型的超參數可以在檔案內調整，已訓練的模型和報表將會產生在`models/`對應名稱的目錄內的文件夾。
- `training_graph_ppo_model.py`：使用 GIN 結合 PPO 的演算法來訓練模型。
- `training_ppo_model.py`：使用 PPO 的演算法來訓練模型。
### 評估模型
您可以執行在`run_scripts/`目錄下帶有`evaluation`開頭的檔案，將會使用對應演算法的已訓練模型去預測魔塔遊戲。
評估模型的超參數必須與訓練時的超參數相同。
- `evaluation_graph_ppo_model.py`：評估 GIN 結合 PPO 演算法的已訓練模型。
- `evaluation_ppo_model.py`：評估 PPO 演算法的已訓練模型。
### Class
- Mota：建立魔塔遊戲環境，可供代理人進行訓練或是預測。
- MotaBuilder：創建魔塔遊戲環境資料，可以自訂義魔塔地圖。
- MotaGenerator：生成隨機的魔塔遊戲環境資料。
### Folder
- algorithms： 深度強化學習的演算法和神經網路。
- env：與魔塔遊戲環境相關的程式。
- models：由深度強化學習所產生已訓練模型和報表。
- run_scripts：供使用者運行的各種腳本，包含訓練和預測。
- tests：程式測試腳本。

## 單元測試 Unit Test
- 本專案使用 pytest 作為測試框架，並使用 coverage.py 收集測試覆蓋率。
- 下列指令將運行所有測試，並生成覆蓋率數據。
```bash
coverage run --omit="tests/*" -m pytest
```
- 下列指令將覆蓋率數據轉換成 html 的報表。  
您可以開啟 `./tests/coverage/index.html` 來檢視程式碼覆蓋狀況。  
```bash
coverage html --skip-empty -d ./tests/coverage
```
- 下列指令將生成覆蓋率徽章（badge）。  
```bash
coverage-badge -o coverage.svg -f
```
- 您若是在 Windows 作業系統下運行，可以參考 `coverage.ps1` 檔案的指令。
- 使用 PyCharm 運行測試時，請至工具列 File -> Settings -> Tools -> Python Integrated Tools ->Testing -> Default test runner，將設定切換成 pytest。
- 測試覆蓋率報表請 [點擊此連結](https://hung61601.github.io/motaDRL/tests/coverage/)。

## 錯誤報告 Bug Report
運行本專案時遇到任何問題，可以透過以下方式聯絡作者：
- Email: hung61601@gmail.com
