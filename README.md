# **Perceptron**

## **How it Works**

1. **Inputs**: The perceptron receives a set of features as inputs, represented as a vector \( \mathbf{x} = [x_1, x_2, ..., x_n] \).
2. **Weights**: Each input feature is associated with a weight \( w_1, w_2, ..., w_n \), determining its importance in the classification decision.
3. **Bias**: A bias term \( b \) is added to shift the decision boundary (the separating line or hyperplane).
4. **Activation Function**: The perceptron computes a weighted sum of inputs, passes it through a step function (also called a threshold function), and outputs a class label (0 or 1).
   - If the weighted sum is greater than a threshold (usually zero), it outputs **1** (positive class).
   - Otherwise, it outputs **0** (negative class).

## **Training the Perceptron**

During training, the perceptron iteratively adjusts its weights to minimize classification errors using the **Perceptron Learning Rule**:
- If the perceptron misclassifies an example, it updates the weights to correct the mistake.
- The update rule is:
  \[
  w = w + \eta (y_{\text{true}} - y_{\text{predicted}}) \mathbf{x}
  \]
  where \( \eta \) is the learning rate, and \( y_{\text{true}} \) and \( y_{\text{predicted}} \) are the true and predicted class labels, respectively.

## **Geometric Interpretation**

The perceptron learns a decision boundary (a hyperplane) that separates the two classes in the feature space. The weights and bias determine the position of this boundary.

## **Limitations**

- The perceptron can only solve problems that are **linearly separable** (where classes can be separated by a straight line or hyperplane).
- It is a **single-layer** model, meaning it cannot solve problems requiring complex, non-linear decision boundaries (such as the XOR problem).

## **Conclusion**

Despite its simplicity, the perceptron is a building block for more complex neural network architectures. It laid the foundation for modern deep learning models like **multi-layer perceptrons (MLPs)** and other advanced neural networks.
