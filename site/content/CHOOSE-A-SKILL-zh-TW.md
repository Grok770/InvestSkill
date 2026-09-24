# 選擇技能

> 24 個框架不算少。本頁將你的*目標*對應到合適的技能——並釐清大家最常問的重疊之處。第一次來？先從 `stock-eval` 開始；它一次就觸及品質、價值與風險。

---

## 從你的目標出發

| 我想要… | 使用 | 接著可考慮 |
|---------|------|-----------|
| 快速篩選一檔股票（要 / 不要） | `stock-eval` | `result-validator` |
| 深入了解這門*生意* | `stock-eval`（其財報層級章節） | `financial-report-analyst` |
| 判斷它*便宜還是貴* | `stock-valuation` | `bear-case` |
| 取得嚴謹的*內在價值* | `stock-valuation`（其 DCF 方法＋敏感度表） | `result-validator` |
| 讀 10-K／10-Q 找出警訊 | `financial-report-analyst` | `earnings-call-analysis` |
| 評估一場法說會 | `earnings-call-analysis` | `options-analysis` |
| 檢查*股利是否安全* | `dividend-analysis` | `portfolio-review` |
| *抓進出場時機* | `technical-analysis` | `chart-master` |
| 評估*競爭護城河* | `competitor-analysis` | `industry-map` |
| 繪製*產業供應鏈*（上游→下游） | `industry-map` | `competitor-analysis` |
| 看*聰明錢*的動向 | `institutional-ownership` | `insider-trading` |
| 追蹤*內部人*買賣 | `insider-trading` | `short-interest` |
| 衡量*軋空*潛力 | `short-interest` | `technical-analysis` |
| 壓力測試*多頭論點*（看下檔風險） | `bear-case` | `stock-eval` |
| 規劃*如何分批建倉*（或管理已套牢的持股） | `position-ladder` | `technical-analysis` |
| 寫下*我為什麼持有*，之後再回頭檢查論點 | `thesis-tracker` | `bear-case` |
| 挑選*選擇權*策略 | `options-analysis` | `technical-analysis` |
| 判讀*總經*環境 | `economics-analysis` | `sector-analysis` |
| 找*類股輪動*機會 | `sector-analysis` | `stock-eval` |
| 檢視我的*整體投組* | `portfolio-review` | `dividend-analysis` |
| 建立*單一完整投資論述* | `full-report` | `result-validator` |
| 匯出*精美的 HTML 報告* | `full-report` | `report-generator` |
| 為報告製作*圖表* | `chart-master` | `report-generator` |
| *快速複核*任何分析 | `result-validator` | — |

---

## 決策樹

```
你的起點是什麼？

├─ 「我有一檔代號，想知道值不值得看」
│     └─ stock-eval  ──(有潛力?)──► full-report ──► result-validator
│
├─ 「我想知道它值多少」
│     └─ stock-valuation（DCF＋同業比較＋EV 倍數＋剩餘收益，三角交叉）
│
├─ 「正值財報季」
│     └─ stock-eval（基準）
│           └─ earnings-call-analysis（法說後）
│                 └─ options-analysis（波動率＋策略）
│
├─ 「我從由上而下／總經思考」
│     └─ economics-analysis ──► sector-analysis ──► stock-eval
│
├─ 「我重視現金流（收息）」
│     └─ dividend-analysis ──► portfolio-review
│
├─ 「我在交易一個型態」
│     └─ short-interest ──► technical-analysis ──► options-analysis ──► chart-master
│
├─ 「我已經持有——該怎麼建倉／管理部位？」
│     └─ stock-eval（論點還成立？）──► technical-analysis（支撐位）
│           └─ position-ladder（階梯、股數區間、賣高買低循環）
│                 └─ thesis-tracker（把論點寫下來；--update 重新檢查）
│
└─ 「我要完整全包、可匯出」
      └─ full-report （全部跑一遍，存成 HTML 檔）
```

---

## 技能比較（大家最常問的重疊）

### `stock-eval` vs. `stock-valuation`
這兩者重疊最多。差別在於**深度與目的**：

| 技能 | 最適合 | 深度 | 產出 |
|------|--------|------|------|
| `stock-eval` | 快速、全面的*要 / 不要*——其深掘章節也負責*了解這門生意* | 先廣，再到財報層級 | 品質＋價值＋護城河＋風險，整合為單一訊號；需要時提供損益表、資產負債表、現金流拆解 |
| `stock-valuation` | *價格合不合理？* | 估值面深入 | P/E · P/S · EV/EBITDA · DCF（含敏感度表）· 剩餘收益，並列比較 |

**經驗法則：** 先 `stock-eval` 判斷*是否*值得深掘；再以 `stock-valuation` 探究*值多少*。單一 DCF 可能精準地錯——`stock-valuation` 用其他方法三角交叉，讓你看清內在價值有多脆弱。

### `position-ladder` vs. `thesis-tracker`
- **`position-ladder`**——*如何*建倉或管理部位：階梯、股數下限／上限、賣高買低循環、洗售警示。它有論點破損閘門，但不會記住論點本身。
- **`thesis-tracker`**——*為什麼*持有：以附門檻的 KPI 與失效觸發條件寫下論點，存成檔案。`--update` 以新資料重新檢查，回傳 INTACT／WEAKENED／BROKEN。

兩者搭配使用：`thesis-tracker` 決定還能不能加碼；`position-ladder` 決定在什麼價位、加多少。

### `catalyst-calendar` vs. `thesis-tracker`
`catalyst-calendar` 列出未來 90 天可能推動股價的事件與日期。`thesis-tracker` 匯入這些日期，並在每個事件後問一個更窄的問題：*這件事是確認、還是削弱了我持有的理由？*

### 別名——`fundamental-analysis`、`dcf-valuation`、`research-bundle`
這三個仍可使用，但屬於**轉址**，不是獨立框架，也不計入 24 個框架：

| 別名 | 現在位於 | 為何合併 |
|------|----------|----------|
| `fundamental-analysis` | `stock-eval` | 只做財報分析、不做估值，總是只給出半個結論 |
| `dcf-valuation` | `stock-valuation` | 單獨的 DCF 會精準地錯；現在是三角交叉模型的第一種方法 |
| `research-bundle` | `full-report` | 同一套引擎；`full-report --depth quick / standard / comprehensive` 涵蓋對話內的研究包，並多存一份 HTML 檔 |

### 基本面 vs. 技術面——何時用哪個
| | 基本面（`stock-eval`、`stock-valuation`、`financial-report-analyst`） | 技術面（`technical-analysis`、`chart-master`） |
|---|---|---|
| 回答 | 該持有*什麼*、它*值不值得* | *何時*進出場 |
| 時間框架 | 數月至數年 | 數日至數月 |
| 輸入 | 財務報表、申報文件 | 價格、成交量、指標 |
| 可一起用？ | 可——基本面挑標的，技術面抓時機 |

它們並非對手。[波段交易旅程](use-cases-zh-tw.html) 同時運用兩者：以基本面選股，以技術面抓時機。

---

> **下一步：** [使用情境](use-cases-zh-tw.html) 展示這些串接的完整流程 · [概念](concepts-zh-tw.html) 解釋各項指標 · [資料與準確性](data-and-accuracy-zh-tw.html) 談如何信任產出。

*僅供教育用途。並非投資建議。*
