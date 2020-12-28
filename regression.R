
library(aod)
library(ggplot2)
library("car", lib.loc="~/R/win-library/3.3")
library("mlogit", lib.loc="~/R/win-library/3.3")

#__________------------------------------------------------------------------------------------------------------------------


df2$prev_time_elapsed_serp = as.numeric(df2$prev_time_elapsed_serp)
df2$prev_num_content_distinct = as.factor(df2$prev_num_content_distinct)
df$prev_query_len = as.numeric(df$prev_query_len)
df$prev_num_query_noclicks = as.factor(df$prev_num_query_noclicks)
df$prev_num_bookmarks = as.factor(df$prev_num_bookmarks)
df2$prev_time_elapsed = as.numeric(df2$prev_time_elapsed)


df2$probhelp_difficult_articulate<-relevel(df2$probhelp_difficult_articulate, "0")

model <- glm(probhelp_difficult_articulate ~ prev_time_elapsed, data = df2, family = binomial())
summary(model)
model <- glm(probhelp_difficult_articulate ~ prev_num_content_distinct, data = df2, family = binomial())
summary(model)
model <- glm(probhelp_difficult_articulate ~ prev_num_bookmarks, data = df2, family = binomial())
summary(model)
model <- glm(probhelp_difficult_articulate ~ prev_query_len, data = df2, family = binomial())
summary(model)
model <- glm(probhelp_difficult_articulate ~ prev_time_elapsed_serp, data = df2, family = binomial())
summary(model)
model <- glm(probhelp_difficult_articulate ~ prev_num_query_noclicks, data = df2, family = binomial())
summary(model)

model <- glm(probhelp_difficult_articulate ~ prev_num_content_distinct + prev_num_bookmarks + prev_time_elapsed + prev_time_elapsed + prev_time_elapsed_serp + prev_query_len + prev_num_query_noclicks, data = df2, family = binomial())
summary(model)

modelChi <- model$null.deviance - model$deviance
modelChi
chidf <- model$df.null - model$df.residual
chidf
chisq.prob <- 1 - pchisq(modelChi, chidf)
chisq.prob
R2.hl<-modelChi/model$null.deviance
R2.hl
R.cs <- 1 - exp ((model$deviance - model$null.deviance)/400) 
R.cs


wald.test(b = coef(model), Sigma = vcov(model), Terms = 4:6)
exp(coef(model))
exp(cbind(OR = coef(model), confint(model)))

with(model, null.deviance - deviance)
with(model, df.null - df.residual)
with(model, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
logLik(model)

model$coefficients
confint(model)
confint.default(model)
exp(model$coefficients)
exp(confint(model))

vif(model)
1/vif(model)

df2$loggpa <- log(df2$prev_time_elapsed)*df2$prev_time_elapsed #numeric only
#df2$logre <- log(df2$gre)*df2$
#df1$logrank <- log(df1$rank)*df1$rank

#model2 <- glm(probhelp_difficult_articulate ~ prev_time_elapsed + prev_num_content_distinct + prev_num_bookmarks + loggpa, data=df2, family=binomial())
#summary(model2)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
