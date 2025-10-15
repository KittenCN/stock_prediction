@echo off
REM 一键横向测试所有主流模型结构
setlocal
set TEST_CODE=000004
set MODE=test
set EPOCH=2

REM LSTM
python scripts\predict.py --model lstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM Attention-LSTM
python scripts\predict.py --model attention_lstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM BiLSTM
python scripts\predict.py --model bilstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM TCN
python scripts\predict.py --model tcn --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM Hybrid
python scripts\predict.py --model hybrid --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM PTFT + VSSM
python scripts\predict.py --model ptft_vssm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM MultiBranch
python scripts\predict.py --model multibranch --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM Transformer
python scripts\predict.py --model transformer --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM CNNLSTM
python scripts\predict.py --model cnnlstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
endlocal
