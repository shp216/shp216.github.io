---
layout: single
title: "Vanilla GAN"
categories: GAN
tag: [GAN]
toc: true



---

## Vanilla GAN

The basic model that expresses GAN

![Vanilla_GAN](../../images/2023-06-30-VanillaGAN/Vanilla_GAN.png)

### Model

Using nn.Linear in Model and to limit values 0~1, using ```sigmoid``` and ```tanh``` function (+ ```denorm```)

```python
Discriminator = nn.Sequential(
            		nn.Linear(image_size, 256),
            		nn.LeakyReLU(0.2),
            		nn.Linear(256, 256),
            		nn.LeakyReLU(0.2),
            		nn.Linear(256, 1),
            		nn.Sigmoid(),
        				)

		Generator = nn.Sequential(
            		nn.Linear(latent_size, 256),
            		nn.ReLU(),
            		nn.Linear(256, 256),
            		nn.BatchNorm1d(256),
            		nn.ReLU(),
            		nn.Linear(256, image_size),
            		nn.Tanh()
        				)
```

### Optimizer

Using Adam Optimizer

```python
self.G_optimizer =
torch.optim.Adam(self.G.parameters(), self.g_lr)
self.D_optimizer =
torch.optim.Adam(self.D.parameters(), self.d_lr)
```

### Loss Function

![loss_func](../../images/2023-06-30-VanillaGAN/loss_func.png)

Discriminator learns to decide ``D(x)->1``, `` D(G(z))->0``

```python
outputs = self.D(images)
d_loss_real = self.criterion(outputs, real_labels)

outputs = self.D(self.G(z))
d_loss_fake = self.criterion(outputs, fake_labels)

d_loss = d_loss_real + d_loss_fake
```

Generator learns to decide ``D(G(z))->1``

```python
outputs = self.D(self.G(z))
g_loss = self.criterion(outputs, real_labels)
```

### Results

![Generated_img](../../images/2023-06-30-VanillaGAN/Generated_img.png)



### Jensen-Shenen Divergence

$ V(G, D) = E_ {x \sim p_ {data} (x)} [ \log(D(x)) ] + E_ {z \sim p_ {z} (z)} [ \log(1-D(G(z))) ] $



