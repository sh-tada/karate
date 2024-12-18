import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def batched_model1(data):
    # バッチ次元に対応するplateを定義
    batch_size = data.shape[0]
    with numpyro.plate("batch", batch_size):
        # 各データセットごとにパラメータを定義
        mu = numpyro.sample("mu", dist.Normal(0, 10))  # 平均
        sigma = numpyro.sample("sigma", dist.LogNormal(0, 1))  # 標準偏差

        # mu, sigma を (batch_size, num_samples) の形状にブロードキャスト
        mu = mu[:, None]  # (batch_size, 1)
        sigma = sigma[:, None]  # (batch_size, 1)

        # 観測データに対する尤度
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)


def batched_model2(data):
    # バッチ次元に対応するplateを定義
    batch_size = data.shape[0]
    with numpyro.plate("batch", batch_size):
        # 各データセットごとにパラメータを定義
        mu = numpyro.sample("mu", dist.Normal(0, 10))  # 平均
        sigma = numpyro.sample("sigma", dist.LogNormal(0, 1))  # 標準偏差

    # mu, sigma を (batch_size, num_samples) の形状にブロードキャスト
    mu = mu[:, None]  # (batch_size, 1)
    sigma = sigma[:, None]  # (batch_size, 1)
    # 観測データに対する尤度
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)


# 複数のデータセットの準備
rng_key = jax.random.PRNGKey(0)
data1 = jax.random.normal(rng_key, (100,)) + 2.0  # 平均2のデータセット
data2 = jax.random.normal(rng_key, (100,)) - 1.0  # 平均-1のデータセット

# データをバッチ化（形状: (batch_size, num_samples)）
batched_data = jnp.stack([data1, data2], axis=0)

# HMC (NUTSアルゴリズム) を定義
kernel = NUTS(batched_model1)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)

# 推定の実行
mcmc.run(rng_key, data=batched_data)

# 結果の取得
mcmc.print_summary()

samples = mcmc.get_samples()
print("mu shape:", samples["mu"].shape)
print("sigma shape:", samples["sigma"].shape)

# HMC (NUTSアルゴリズム) を定義
kernel = NUTS(batched_model2)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)

# 推定の実行
mcmc.run(rng_key, data=batched_data)

# 結果の取得
mcmc.print_summary()

samples = mcmc.get_samples()
print("mu shape:", samples["mu"].shape)
print("sigma shape:", samples["sigma"].shape)
