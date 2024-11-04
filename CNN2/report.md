# CNN 2

## Conduct experiments with different neural network architectures:

### No convolutional layers

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | fc1 | Linear | 393 K | train
1 | fc2 | Linear | 1.3 K | train
2 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
394 K Trainable params
0 Non-trainable params
394 K Total params
1.579 Total estimated model params size (MB)
3 Modules in train mode
0 Modules in eval mode

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.5033000111579895
test_loss 1.435502290725708
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Different normalization types

#### Batch Normalization

##### Architecture

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | conv1 | Conv2d | 448 | train
1 | bn1 | BatchNorm2d | 32 | train
2 | conv2 | Conv2d | 4.6 K | train
3 | bn2 | BatchNorm2d | 64 | train
4 | fc1 | Linear | 802 K | train
5 | fc2 | Linear | 1.3 K | train
6 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
809 K Trainable params
0 Non-trainable params
809 K Total params
3.238 Total estimated model params size (MB)
7 Modules in train mode
0 Modules in eval mode
```

##### Results

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.6891999840736389
test_loss 0.9022014737129211
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

#### Layer Normalization

##### Architecture

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | conv1 | Conv2d | 448 | train
1 | norm1 | LayerNorm | 28.8 K | train
2 | conv2 | Conv2d | 4.6 K | train
3 | norm2 | LayerNorm | 50.2 K | train
4 | fc1 | Linear | 802 K | train
5 | norm3 | LayerNorm | 256 | train
6 | fc2 | Linear | 1.3 K | train
7 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
888 K Trainable params
0 Non-trainable params
888 K Total params
3.554 Total estimated model params size (MB)
8 Modules in train mode
0 Modules in eval mode
```

##### Results

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.6858999729156494
test_loss 1.1074416637420654
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

#### Instance Normalization

##### Architecture

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | conv1 | Conv2d | 448 | train
1 | norm1 | InstanceNorm2d | 0 | train
2 | conv2 | Conv2d | 4.6 K | train
3 | norm2 | InstanceNorm2d | 0 | train
4 | fc1 | Linear | 802 K | train
5 | fc2 | Linear | 1.3 K | train
6 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
809 K Trainable params
0 Non-trainable params
809 K Total params
3.237 Total estimated model params size (MB)
7 Modules in train mode
0 Modules in eval mode
```

##### Results

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.6593999862670898
test_loss 1.0029935836791992
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

#### Group Normalization

##### Architecture

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | conv1 | Conv2d | 448 | train
1 | gn1 | GroupNorm | 32 | train
2 | conv2 | Conv2d | 4.6 K | train
3 | gn2 | GroupNorm | 64 | train
4 | fc1 | Linear | 802 K | train
5 | fc2 | Linear | 1.3 K | train
6 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
809 K Trainable params
0 Non-trainable params
809 K Total params
3.238 Total estimated model params size (MB)
7 Modules in train mode
0 Modules in eval mode
```

##### Results

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.6802999973297119
test_loss 0.9251918792724609
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Dropout

#### Architecture

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | conv1 | Conv2d | 448 | train
1 | bn1 | BatchNorm2d | 32 | train
2 | conv2 | Conv2d | 4.6 K | train
3 | bn2 | BatchNorm2d | 64 | train
4 | fc1 | Linear | 802 K | train
5 | dropout | Dropout | 0 | train
6 | fc2 | Linear | 1.3 K | train
7 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
809 K Trainable params
0 Non-trainable params
809 K Total params
3.238 Total estimated model params size (MB)
8 Modules in train mode
0 Modules in eval mode
```

#### Results

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.6863999962806702
test_loss 0.9021387696266174
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Different epochs [5, 10, 15, 20]

#### Architecture

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | conv1 | Conv2d | 448 | train
1 | conv2 | Conv2d | 4.6 K | train
2 | norm1 | BatchNorm2d | 32 | train
3 | norm2 | GroupNorm | 64 | train
4 | dropout | Dropout | 0 | train
5 | fc1 | Linear | 802 K | train
6 | fc2 | Linear | 1.3 K | train
7 | norm_fc | LayerNorm | 256 | train
8 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
809 K Trainable params
0 Non-trainable params
809 K Total params
3.239 Total estimated model params size (MB)
9 Modules in train mode
0 Modules in eval mode
```

#### Results

##### 5 epochs

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.7062000036239624
test_loss 0.8403913378715515
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

##### 10 epochs

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.7111999988555908
test_loss 0.8998528122901917
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

##### 15 epochs

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.70169997215271
test_loss 1.101760983467102
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

##### 20 epochs

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.6884999871253967
test_loss 1.39614737033844
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Different features

##### Architecture

```markdown
  | Name | Type | Params | Mode
--------------------------------------------------------
0 | conv1 | Conv2d | 3.6 K | train
1 | conv2 | Conv2d | 73.8 K | train
2 | norm1 | BatchNorm2d | 128 | train
3 | dropout | Dropout | 0 | train
4 | fc1 | Linear | 663 K | train
5 | fc2 | Linear | 1.3 K | train
6 | norm_fc | LayerNorm | 256 | train
7 | accuracy | MulticlassAccuracy | 0 | train
--------------------------------------------------------
742 K Trainable params
0 Non-trainable params
742 K Total params
2.971 Total estimated model params size (MB)
8 Modules in train mode
0 Modules in eval mode
```

#### More features in Convolutional Layers (128 output channels)

```markdown

##### Results after 5 epochs

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.73089998960495
test_loss 0.8028696179389954
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

#### More features in Fully Connected Layers (256 output features)

##### Results after 5 epochs

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.7558000087738037
test_loss 0.8004845380783081
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

#### 32 output features in Convolutional Layers

##### Results after 5 epochs

```markdown
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Test metric DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
test_accuracy 0.7258999943733215
test_loss 0.843691885471344
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

## Q&A

### Where should the normalization layer be in the neural network?

A normalization layer, like Batch Normalization, is typically placed after the fully connected or convolutional layers
and before the activation function. This helps in stabilizing and speeding up training by standardizing the inputs to a
layer.

### Where should the dropout layer be in the neural network?

A dropout layer is usually placed after the activation functions of the layers where you want to prevent overfitting.
This could be after fully connected layers but before the output layer during training. This helps in regularizing the
model by randomly setting a fraction of the input units to zero during training. By doing so the model learns to be more
robust and generalizes better to unseen data.