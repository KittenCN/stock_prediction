# stock_prediction

鍩轰簬 PyTorch 鐨勮偂绁ㄤ环鏍奸娴嬪疄楠岄」鐩紝鎻愪緵浠庢暟鎹噰闆嗐€侀澶勭悊鍒版ā鍨嬭缁冧笌鎺ㄧ悊鐨勫叏娴佺▼绀轰緥銆傚綋鍓嶉噸鐐归獙璇佸灏哄害娣峰悎妯″瀷锛圱emporalHybridNet锛変互鍙?PTFT + Variational SSM 鍙岃建缁勫悎銆?

## 鍔熻兘姒傝
- **琛屾儏閲囬泦**锛氬皝瑁?`tushare`/`akshare`/`yfinance` 涓夌被鏁版嵁婧愩€?
- **鏁版嵁棰勫鐞?*锛氬皢鏃ョ嚎 CSV 鑱氬悎涓?`pkl_handle/train.pkl` 闃熷垪锛屾敮鎸侀噸澶嶅姞杞姐€?
- **鐗瑰緛宸ョ▼**锛?
  feature_engineering.py 鑷姩鐢熸垚瀵规暟鏀剁泭涓庡樊鍒嗙壒寰侊紝鎸夐厤缃悎骞跺畯瑙傘€佽涓氥€佽垎鎯呭鐢熷彉閲忥紝鏀寔 per-symbol 褰掍竴 (enable_symbol_normalization) 骞剁紦瀛樼粺璁￠噺锛涘惎鐢?use_symbol_embedding 鏃朵細杈撳嚭鑲＄エ ID 绱㈠紩锛屾ā鍨嬪彲鍏变韩鍙涔犵殑宓屽叆鍚戦噺銆?
- **模型训练**：`src/stock_prediction/train.py` 提供统一入口，可选择 LSTM、Transformer、TemporalHybridNet、PTFT_VSSM、Diffusion、Graph 等模型，内置 Trainer/LR Scheduler/Early Stopping；Hybrid 默认使用 `HybridLoss`（MSE+分位+方向+Regime）并叠加波动/极值/均值/收益约束（默认权重分别为 0.12/0.02/0.05/0.08）。
- **Hybrid 鎬荤嚎**锛?-model hybrid 鑱氬悎鍗风Н/GRU/Attention 涓?PTFT銆乂SSM銆丏iffusion銆丟raph 鍒嗘敮杈撳嚭锛屽彲閫氳繃 ranch_config 璁剧疆鍒嗘敮鍏堥獙鏉冮噸锛岃蒋闂ㄦ帶鑷姩璋冭妭璐＄尞搴︼紝骞跺湪澶氳偂绁ㄥ満鏅笅鍏变韩鑲＄エ宓屽叆銆?
- **鎺ㄧ悊棰勬祴**锛歴rc/stock_prediction/predict.py 璐熻矗鍔犺浇妯″瀷鏉冮噸骞惰緭鍑洪娴嬬粨鏋滐紝淇濇寔涓庤缁冮樁娈典竴鑷寸殑鐗瑰緛鍔犲伐涓庡祵鍏ョ瓥鐣ャ€?
- **评估指标**：`metrics.py` 自动采集 RMSE、MAPE、分位覆盖率、VaR、CVaR 等金融指标，并输出振幅/均值诊断（distribution_report），训练/测试后保存至 `output/metrics_*.json`。
- **预测诊断**：`scripts/analyze_predictions.py` 可批量扫描 `png/test` 或 `png/predict` 下的 CSV，标记振幅塌缩（std_ratio）与均值偏移（bias）等问题，便于回归审查。
- **鎶€鏈寚鏍囧簱**锛歚target.py` 鍐呯疆甯歌鎸囨爣锛圡ACD銆並DJ銆丏MI銆丄TR 绛夛級銆?
  
  褰掍竴鍖栧弬鏁版寔涔呭寲涓庤嚜鍔ㄥ洖濉細
  - 璁粌淇濆瓨妯″瀷鏃舵€讳細鍐欏嚭涓庢潈閲嶅悓鍚嶇殑 `*_norm_params*.json`锛堝寘鍚?mean_list/std_list/show_list/name_list锛夈€?
  - 鑻ュ湪 PKL 妯″紡涓嬪叏灞€鍧囧€?鏂瑰樊鍒楄〃涓虹┖锛屽皢鑷姩浠?`pkl_handle/train.pkl` 璁＄畻骞跺啓鍏ワ紙鏃犻渶鎵嬪伐鑴氭湰锛夈€?
  - test()/predict() 鍦ㄥ姞杞芥潈閲嶅墠浼氫紭鍏堣鍙栧搴旂殑 `*_norm_params*.json`锛岀‘淇濆弽褰掍竴鍖栨纭€?

## 蹇€熷紑濮?
```bash
conda activate stock_prediction
pip install -r requirements.txt
python scripts/train.py --mode train --model ptft_vssm --pkl 1 --epoch 2
# 鎺ㄧ悊绀轰緥
python scripts/predict.py --model ptft_vssm --test_code 000001
```
> 棣栨杩愯璇峰厛鎵ц `python scripts/getdata.py --api akshare --code 000001.SZ` 涓?`python scripts/data_preprocess.py --pklname train.pkl` 鍑嗗鏁版嵁銆?

## 鐩綍缁撴瀯
```
project-root/
鈹溾攢 src/stock_prediction/
鈹? 鈹溾攢 config.py              # 璺緞涓庣洰褰曠鐞?
鈹? 鈹溾攢 init.py                # 瓒呭弬鏁般€佽澶囦笌鍏变韩闃熷垪
鈹? 鈹溾攢 common.py              # 鏁版嵁闆嗐€佸彲瑙嗗寲銆佹ā鍨嬩繚瀛樺伐鍏?
鈹? 鈹溾攢 models/                # 妯″瀷闆嗗悎锛圠STM/Transformer/TemporalHybridNet/PTFT_VSSM 绛夛級
鈹? 鈹溾攢 train.py               # 璁粌 / 娴嬭瘯涓绘祦绋?
鈹? 鈹溾攢 predict.py             # 鎺ㄧ悊鍏ュ彛
鈹? 鈹溾攢 getdata.py             # 琛屾儏閲囬泦鑴氭湰
鈹? 鈹溾攢 data_preprocess.py     # 棰勫鐞嗕笌搴忓垪鍖?
鈹? 鈹斺攢 target.py              # 鎶€鏈寚鏍囧嚱鏁板簱
鈹溾攢 scripts/
鈹? 鈹溾攢 train.py               # 鍛戒护琛岃缁冨皝瑁?
鈹? 鈹斺攢 predict.py             # 鍛戒护琛屾帹鐞嗗皝瑁?
鈹溾攢 tests/                    # PyTest 鐢ㄤ緥
鈹溾攢 docs/                     # 鏋舵瀯銆佺瓥鐣ャ€佽繍缁存枃妗?
鈹溾攢 models/                   # 璁粌鍚庣殑妯″瀷鏉冮噸
鈹溾攢 stock_daily/              # 鍘熷琛屾儏鏁版嵁
鈹溾攢 pkl_handle/               # 棰勫鐞嗗悗鐨勯槦鍒楁枃浠?
鈹斺攢 CHANGELOG.md              # 鍙樻洿璁板綍
```

## 鏀寔妯″瀷 (`--model`)
| 鍙傛暟 | 缁撴瀯姒傝堪 | 鍦烘櫙 |
| ---- | -------- | ---- |
| `lstm` | 3 灞?LSTM + 鍏ㄨ繛鎺?| 鍩虹嚎楠岃瘉 |
| `attention_lstm` | LSTM + 娉ㄦ剰鍔?| 鍏虫敞鍏抽敭鏃堕棿鐗囨 |
| `bilstm` | 鍙屽悜 LSTM | 鍔犲己涓婁笅鏂囧缓妯?|
| `tcn` | 鏃跺簭鍗风Н缃戠粶 | 鎹曟崏灞€閮ㄦā寮?|
| `multibranch` | 浠锋牸/鎸囨爣鍙屽垎鏀?LSTM | 闈㈠悜澶氱壒寰佹棌 |
| `transformer` | 鑷畾涔?Transformer 缂栬В鐮佺粨鏋?| 闀垮簭鍒楀缓妯?|
| `cnnlstm` | CNN + LSTM + Attention | 澶氭棰勬祴锛堥渶 `predict_days` > 0锛墊
| `hybrid` | Hybrid Aggregator锛堝嵎绉?GRU + PTFT/VSSM/Diffusion/Graph 鎬荤嚎锛墊 澶氭ā鎬佺壒寰佽瀺鍚?|
| `ptft_vssm` | PTFT + Variational SSM 鍙岃建缁勫悎 | 姒傜巼棰勬祴涓庨闄╄瘎浼?|
| `diffusion` | DiffusionForecaster锛堟墿鏁ｅ紡鍘诲櫔瑙ｇ爜锛?| 鎯呮櫙鐢熸垚銆佸熬閮ㄩ闄╁垎鏋?|
| `graph` | GraphTemporalModel锛堣嚜閫傚簲鍥剧粨鏋勶級 | 澶氳祫浜у叧鑱斿缓妯?|

鎵归噺瀵规瘮鍙墽琛?`scripts\run_all_models.bat`锛堥粯璁よ繍琛岃缁?娴嬭瘯妯″紡锛夈€?

## 鐗瑰緛宸ョ▼閰嶇疆
- 鎵€鏈夌壒寰佸姞宸ョ瓥鐣ョ敱 `config/config.yaml` 鐨?`features` 鑺傚畾涔夛細
  - `target_mode=log_return`銆乣return_kind=log`锛氶粯璁ゅ鏀剁洏浠风敓鎴愬鏁版敹鐩婄巼鏍囩銆?
  - `difference_columns`銆乣volatility_columns`锛氭帶鍒跺樊鍒嗕笌婊戝姩绐楀彛缁熻瀛楁銆?
  - `external_sources`锛氱櫧鍚嶅崟寮忓紩鍏ュ畯瑙?琛屼笟/鑸嗘儏 CSV锛屾寜 `trade_date` 瀵归綈骞舵敮鎸佸墠鍚戝～鍏呫€?
  - `multi_stock: true`锛氶粯璁ゅ湪澶氳偂绁ㄥ満鏅仛鍚堣缁冩牱鏈紝鑷姩鐢熸垚鏂瑰悜鏍囩銆?
- 鑻ヨ嚜瀹氫箟澶栫敓鐗瑰緛锛屼繚鎸佹棩鏈熷垪涓哄叓浣嶅瓧绗︿覆锛堝 `20241203`锛夛紝骞舵斁缃湪 `config/external/` 涓嬪嵆鍙€?

## 甯哥敤鍛戒护
```bash
# 鎶撳彇琛屾儏
python scripts/getdata.py --api akshare --code 000001.SZ

# 鏁版嵁棰勫鐞?
python scripts/data_preprocess.py --pklname train.pkl

# 璁粌绀轰緥
python scripts/train.py --mode train --model transformer --epoch 2

# 鎺ㄧ悊绀轰緥
python scripts/predict.py --model transformer --test_code 000001 --predict_days 3

# 杩愯娴嬭瘯
pytest -q

# 鏌ョ湅璁粌鎸囨爣锛堣缁冨悗鑷姩鐢熸垚锛?
cat output/metrics_*.json
```

## 娴嬭瘯涓庤川閲?
- `pytest`锛氳鐩栫壒寰佸伐绋嬨€丷egime 鑷€傚簲浠ュ強鑲＄エ宓屽叆鐩稿叧鍗曞厓娴嬭瘯锛圖iffusion/Graph/Hybrid/PTFT 绛夛級锛屽綋鍓?30+ 椤瑰潎閫氳繃銆?
- 寤鸿鍦ㄦ彁浜ゅ墠鎵ц `pytest -q`锛屽苟鎸夐渶杩愯 `ruff` / `black` / `mypy`銆?
- 鏆撮湶 `create_predictor()` 甯姪鑴氭湰鎴栨祴璇曞揩閫熸瀯閫犳帹鐞嗗櫒銆?

## 鏂囨。绱㈠紩
- `docs/system_design.md`锛氭灦鏋勬嫇鎵戜笌鍏抽敭鍐崇瓥
- `docs/model_strategy.md`锛氭ā鍨嬫柟妗堣璁′笌鎺ㄨ崘缁勫悎
- `docs/user_guide.md`锛氬懡浠よ/妯″潡鐢ㄦ硶銆佽繍缁翠笌澶氭ā鍨嬫祴璇?
- `docs/maintenance.md`锛氱粨鏋勮皟鏁淬€佷慨澶嶈褰曚笌鏀硅繘寤鸿

## 甯歌闂
| 闂 | 鍘熷洜 | 瑙ｅ喅鏂瑰紡 |
| ---- | ---- | -------- |
| `queue.Queue` 鍙嶅簭鍒楀寲鎶ラ敊 | Python 3.13+ 灞炴€у彉鏇?| 宸插湪 `common.ensure_queue_compatibility()` 鍏滃簳锛涘繀瑕佹椂閲嶆柊鐢熸垚 `train.pkl` |
| 瀵煎叆瑙﹀彂 `SystemExit` | CLI 鍦ㄥ鍏ラ樁娈佃В鏋愬懡浠よ鍙傛暟 | `predict.py` / `train.py` 宸叉敼涓洪粯璁ゅ弬鏁板璞★紝鍙洿鎺ュ鍏?|
| CPU 妯″紡鍑虹幇 AMP 鎻愮ず | GradScaler 榛樿閽堝 CUDA | 鎺ㄧ悊涓庤缁冧細鑷姩闄嶇骇锛屽彲蹇界暐鎴栧叧闂?AMP |
| 妯″瀷淇濆瓨闃诲 | weight_norm 涓庢繁鎷疯礉鍐茬獊 | `thread_save_model` 宸叉敼涓轰繚瀛?state_dict 骞惰縼绉诲埌 CPU |
| 褰掍竴鍖栧弬鏁版枃浠朵负绌猴紙鏃фā鍨嬶級 | 鏃╂湡鐗堟湰鏈繚瀛?璁＄畻 mean/std | 瀵瑰巻鍙叉ā鍨嬫墽琛屼竴娆?`python scripts\fix_norm_params.py`锛涙柊璁粌鐨勬ā鍨嬪湪淇濆瓨鏃朵細鑷姩浠?PKL 鍥炲～骞跺啓鍑?`*_norm_params*.json` |
| 鎺ㄧ悊鏃?30 vs 46 缁村害涓嶅尮閰?| 鍚敤 symbol embedding 鍚庨渶瑕?`_symbol_index` | 璁粌/娴嬭瘯/鎺ㄧ悊宸茬粺涓€鍦ㄦ暟鎹绾夸腑娉ㄥ叆 symbol 绱㈠紩锛涚‘淇?predict/test 璇诲彇鍒颁簡 `*_Model_args.json` 浠ュ鐜拌缁冮厤缃?|

## 璐＄尞鎸囧崡
1. 鏂板妯″瀷璇峰湪 `src/stock_prediction/models/` 涓疄鐜帮紝骞跺湪璁粌/鎺ㄧ悊鍏ュ彛娉ㄥ唽銆?
2. 鍚屾鏇存柊娴嬭瘯锛坄tests/test_models.py`锛変笌鏂囨。锛堝挨鍏舵槸 `docs/model_strategy.md`銆乣CHANGELOG.md`锛夈€?
3. 閬靛惊缂栫爜瑙勮寖锛圥EP8 + 绫诲瀷娉ㄩ噴锛夛紝鎻愪氦鍓嶈杩愯 `pytest`銆?

---
鏇村鑳屾櫙涓庢湭鏉ヨ鍒掞紝璇峰弬闃?`docs/model_strategy.md` 涓?`docs/system_design.md`銆?

