# 프로그래머를 위한 베이지안 with 파이썬
import numpy as np
import scipy
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import seaborn as sns

# 1.4. 컴퓨터를 사용하여 베이지안 추론하기
count_data = np.loadtxt("practice/data/txtdata.csv")
n_count_data = len(count_data)

# Parameter
# create pymc3 variables corresponding to lambdas
# assign them to pymc3's stochastic variables
# 관련 변수들을 model 안에 모두 집어 넣음 (attribute로 call 가능)
with pm.Model() as model:
    alpha = 1.0 / count_data.mean()  # Recall count_data is the
    # variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)

print("Random Output:", tau.random(), tau.random())

with model:
    idx = np.arange(n_count_data)
    # switch: assign lambda1 or lambda2 as the value of lambda_
    # depending on what side of tau we are on
    # lambda1 ~ tau ~ lambda2
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
    # 데이터 확인 obs.observations
    obs = pm.Poisson('obs', lambda_, observed=count_data)


# Best: Bayesian Estimation Supersedes T-test
A = np.random.normal(30, 4, size=1000)
B = np.random.normal(26, 7, size=1000)

# Prior
# 1) mu_A, mu_B Prior: 정규 분포
# 2) std_A, std_B Prior: 균일 분포
# 3) nu_minus_1: 자유도 v의 분포: 이동된 지수 분포
pooled_mean = np.r_[A, B].mean()
pooled_std = np.r_[A, B].std()

# 만약 mu_A, mu_B에 대한 사전 지식이 없다면 무정보 사전 분포를 정의하는 것이 좋음
# 정규 분포의 표준 편차에 1000 같은 큰 숫자를 곱해주자.
# 표준편차 역시 Lower, Upper Bound를 크게 해주자.
tau = 1 / (1000*pooled_std**2)    # Precision Parameter

with pm.Model() as model:
    mu_A = pm.Normal("mu_A", pooled_mean, tau)
    mu_B = pm.Normal("mu_B", pooled_mean, tau)
    std_A = pm.Uniform("std_A", pooled_std/1000, 1000*pooled_std)
    std_B = pm.Uniform("std_B", pooled_std/1000, 1000*pooled_std)
    nu_minus_1 = pm.Exponential("nu_1", 1/29)

    # Likelihood: Noncentral T-distribution
    obs_A = pm.distributions.continuous.StudentT("obs_A", observed=A, nu=nu_minus_1 + 1, mu=mu_A, lam=1 / std_A ** 2)
    obs_B = pm.distributions.continuous.StudentT("obs_B", observed=B, nu=nu_minus_1 + 1, mu=mu_B, lam=1 / std_B ** 2)

    # MCMC
    step = pm.Metropolis([obs_A, obs_B, mu_A, mu_B, std_A, std_B, nu_minus_1])
    trace = pm.sampling.sample(draws=20000, step=step)
    burned_trace = trace[10000:]

trace_df = pm.trace_to_dataframe(burned_trace)

# Result
mu_A_trace = trace_df["mu_A"].values
mu_B_trace = trace_df["mu_B"].values
std_A_trace = trace_df["std_A"].values
std_B_trace = trace_df["std_A"].values #[:]: trace object => ndarray
nu_trace = trace_df["nu_1"].values + 1


def _hist(data,label,**kwargs):
    return plt.hist(data,bins=40,histtype="stepfilled",alpha=.95,label=label, **kwargs)

ax = plt.subplot(3,1,1)
_hist(mu_A_trace,"A")
_hist(mu_B_trace,"B")
plt.legend ()
plt.title("Posterior distributions of $\mu$")


