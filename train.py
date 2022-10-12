from utils.utils import get_flying_chairs_data_paths, make_dataset_split
from torch.utils.data import DataLoader
from networks.FlowNetSimple import FlowNetSimple
from torchvision import transforms
from utils.CustomDataset import CustomDataset
import pytorch_lightning as pl

flyingChairsData = get_flying_chairs_data_paths('/media/nehil/nehil/flyingChairsDataset/FlyingChairs_release/data')

train_set, test_set = make_dataset_split(flyingChairsData, split=.9)
train_set, val_set = make_dataset_split(train_set)

# Report split sizes
print('Whole dataset has {} instances'.format(len(flyingChairsData)))
print('Training set has {} instances'.format(len(train_set)))
print('Validation set has {} instances'.format(len(val_set)))
print('Test set has {} instances'.format(len(test_set)))

'''
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
img = cv2.imread(train_set[0][0][0])
# Remember, opencv by default reads images in BGR rather than RGB
# So we fix that by the following
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# Now, for small images like yours or any similar ones we use for example purpose to understand image processing operations or computer graphics
# Using opencv's cv2.imshow()
# Or google.colab.patches.cv2_imshow() [in case we are on Google Colab]
# Would not be of much use as the output would be very small to visualize
# Instead using matplotlib.pyplot.imshow() would give a decent visualization

print(img.shape)
plt.imshow(img)

plt.show();'''


train_data = CustomDataset(train_set)
val_data = CustomDataset(val_set)
train_loader = DataLoader(train_data, batch_size=8, num_workers=8)
val_loader = DataLoader(val_data, batch_size=8, num_workers=8)

# model
model = FlowNetSimple()

# training
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader, val_loader)