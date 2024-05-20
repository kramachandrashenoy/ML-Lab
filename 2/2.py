'''Scatter Plot
Scatter plots are useful for visualizing relationships between two variables.'''

import matplotlib.pyplot as plt
import numpy as np

# Generate random data
x = np.random.rand(100)
y = np.random.rand(100)

# Create scatter plot
plt.scatter(x, y, c='blue', alpha=0.5)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()


'''Box Plot
Box plots are useful for visualizing the distribution of data and identifying outliers.'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate random data
data = np.random.randn(100)

# Create box plot
sns.boxplot(data=data)
plt.title('Box Plot')
plt.show()



'''Heat Map
Heat maps are useful for visualizing data in matrix form.'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.rand(10, 12)

# Create heat map
sns.heatmap(data, annot=True, cmap='viridis')
plt.title('Heat Map')
plt.show()



'''Contour Plot
Contour plots are useful for visualizing 3D data in two dimensions.'''

import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create contour plot
plt.contour(X, Y, Z, levels=20, cmap='RdGy')
plt.title('Contour Plot')
plt.show()



'''3D Surface Plot
3D surface plots are useful for visualizing 3D data.'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D Surface Plot')
plt.show()
