@echo off
REM 一键横向测试所有主流模型结�?
setlocal
set TEST_CODE=000004
set MODE=test
set EPOCH=2

REM LSTM
python scripts\\train.py --model lstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM Attention-LSTM
python scripts\\train.py --model attention_lstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM BiLSTM
python scripts\\train.py --model bilstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM TCN
python scripts\\train.py --model tcn --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM Hybrid
python scripts\\train.py --model hybrid --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM PTFT + VSSM
python scripts\\train.py --model ptft_vssm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM MultiBranch
python scripts\\train.py --model multibranch --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM Transformer
python scripts\\train.py --model transformer --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
REM CNNLSTM
python scripts\\train.py --model cnnlstm --mode %MODE% --test_code %TEST_CODE% --pkl 1 --epoch %EPOCH%
endlocal
