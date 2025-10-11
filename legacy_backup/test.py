import akshare as ak

# stock_comment_detail_zlkp_jgcyd_em_df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol="600975")
# print(stock_comment_detail_zlkp_jgcyd_em_df)

# stock_comment_detail_zhpj_lspf_em_df = ak.stock_comment_detail_zhpj_lspf_em(symbol="600975")
# print(stock_comment_detail_zhpj_lspf_em_df)

stock_hot_follow_xq_df = ak.stock_hot_follow_xq(symbol="最热门")
print(stock_hot_follow_xq_df)

stock_daily_data = ak.stock_zh_a_daily(symbol="sh600975", adjust="qfq")
print(stock_daily_data)
