# AI-Classification

The goal of this assignment is to design and implement a neural network capable of classifying points within a 2D space into four predefined categories (red, green, blue, and purple) based on their X and Y coordinates.
The neural network should learn the underlying patterns from an initial set of 20 points (5 for each class) and subsequently classify new points accurately.

R: [-4500, -4400], [-4100, -3000], [-1800, -2400], [-2500, -3400] a [-2000, -1400] 
G: [+4500, -4400], [+4100, -3000], [+1800, -2400], [+2500, -3400] a [+2000, -1400] 
B: [-4500, +4400], [-4100, +3000], [-1800, +2400], [-2500, +3400] a [-2000, +1400] 
P: [+4500, +4400], [+4100, +3000], [+1800, +2400], [+2500, +3400] a [+2000, +1400]

![image](https://github.com/user-attachments/assets/e30488a1-c1e6-4efa-829b-8240cd9d5215)


After training on this dataset, the neural network will be tested on its ability to accurately classify new points in the 2D space. The success of the neural network will be evaluated by comparing its predictions for the class of each point against the known classes of the 40,000 generated points. This evaluation will provide insights into the effectiveness of the neural network in generalizing from the training data and classifying unseen instances in the 2D space.
For each experiment conducted, it is essential to create visualizations of the resulting 2D surface. This visualization involves colouring the entire area based on the classifications made by the neural network. By visually representing the coloured regions, one can gain insights into how well the neural network has learned and generalized the patterns within the 2D space.
Documentation is a key aspect of this assignment, requiring a comprehensive description of the specific algorithm and data representation employed in the neural network. Documentation should provide a clear and detailed explanation of the chosen neural network architecture, the training process, and any optimizations or techniques utilized. Additionally, it should include insights into the data representation methods employed for both the initial set of points and the newly generated dataset.
