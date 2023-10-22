library(arrow)
library(data.table)

print('Exporting data to RData format')
df <- read_parquet('../data/dashboard/ses.parquet')
# df <- as.data.table(df)
# save(df, file = '../data/dashboard/ses.RData')
