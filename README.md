# ResFGB
This is a Theano(>=1.0.0) implementation of "[Functional gradient boosting based on residual network perception](https://arxiv.org/abs/1802.09031)".

ResFGB is a functional gradient boosting method for learning a resnet-like deep neural network for non-linear classification problems. The model is composed of a linear classifier such as logistic regression and support vector machine, and a feature extraction.
In each iteration, these components are trained by alternate optimization, that is, a linear classifier is trained to classify obtained samples through a feature extraction and this extraction map is updated by stacking a resnet-type layer to move samples along the direction of increasing the linear separability. We finally obtain a highly non-linear classifier forming a residual network.

## Usage
A simple pseudocode is provided below.

__Note__: `(X,Y)`: training data, `(Xv,Yv)`: validation data, `(Xt,Yt)`: test data.
These are numpy arrays.
`n_data`: the number of training data, `input_dim`: dimension of the input space, `n_class`: the number of classes.
__A label set should be an integer sequence starting with zero.__

```python
from resfgb.models import ResFGB, get_hyperparams

hparams = get_hyperparams( n_data, input_dim, n_class )
model = ResFGB( **hparams )
best_iters,_ ,_ = model.fit( X, Y, Xv, Yv, use_best_iter=True )

train_loss, train_acc = model.evaluate( X, Y )
print( 'train_loss: {0}, train_acc: {1}'.format(train_loss, train_acc) )

test_loss, test_acc  = model.evaluate( Xt,  Yt )
print( 'test_loss : {0}, test_acc : {1}'.format(test_loss, test_acc) )
```

See `examples/sample_resfgb.py` for more detail.

## Hyperparameters
Hyperparameters of ResFGB are mainly divided three types: the first is for learning a linear classifier, the second is for learning a multi-layer network as a resblock, and the other is for the functional gradient method.

The hyperparameters are listed below.
'Default' is a value set by the function `resfgb.models.get_hyperparams`.
`input_dim` and `n_class` stand for the dimension of the input space and the number of classes, respectively.

### For the linear model
- `shape`[default=(input\_dim, n\_class)]
	- Shape of the linear model, which __should not be changed__.
- `wr`[default=1/n_data]
	- L2-regularization parameter.
- `bias`[default=True]
	- Flag for whether to include bias term or not.
- `eta`[default=1e-2]
	- Learning rate for Nesterov's momentum method.
- `momentum`[default=0.9]
	- Momentum parameter for Nesterov's momentum method.
- `minibatch_size`[default=100]
	- Minibatch size to compute stochastic gradients.
- `max_epoch`[default=100]
	- The number of epochs for learning a linear model.
- `tune_eta`[default=True]
	- Flag for whether to tune learning rate or not.
- `scale`[default=1.0] 
	- Positive number by which a tuned learning rate is multiplied.
- `eval_iters`[default=1000]
	- The number of iterations in a trial for tuning learning rate.
- `early_stop`[default=10]
	- When the training loss does not improve while this number of epochs, the training is stopped. 

### For the resblock
- `shape`[default=(input_dim,100,100,100,100,input_dim)]
	- Shape of the multi-layer perceptron. __Dimensions of the input and last layer should set to input_dim__.
- `wr`[default=1/n_data]
	- L2-regularization parameter.
- `eta`[default=1e-2]
	- Learning rate for Nesterov's momentum method.
- `momentum`[default=0.9] 
	- Momentum parameter for Nesterov's momentum method.
- `minibatch_size`[default=100]
	- Minibatch size to compute stochastic gradients.
- `max_epoch`[default=50]
	- The number of epochs for learning a linear model.
- `tune_eta`[default=True]
	- Flag for whether to tune learning rate or not.
- `scale`[default=1.0]
	- Positive number by which a tuned learning rate is multiplied.
- `eval_iters`[default=1000]
	- The number of iterations in a trial for tuning learning rate.
- `early_stop`[default=10]
	- When the training loss does not improve while this number of epochs, the training is stopped. 

### For the functional gradient method
- `model_type`[default='logistic']
	- Type of the linear model: 'logistic' or 'smooth_hinge'.
- `model_hparams`[default=model_hparams] 
	- Dictionary of the hyperparameter for the linear model.
- `resblock_hparams`[default=resblock_hparams] 
	- Dictionary of the hyperparameter for the resblock.
- `fg_eta`[default=1e-1]
	- Learning rate used in the functional gradient method.
- `max_iters`[default=30
	- The number of iterations of the functional gradient method, which corresponds to the depth of an obtained network. 
- `seed`[default=1]
   - Random seed used in the method.
