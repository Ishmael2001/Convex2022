import math
import json

class Node(object):
  """
  表示具体的数值或者某个Op的数据结果。
  """
  global_id = -1
  
  def __init__(self, op, inputs):
    self.inputs = inputs # 产生该Node的输入
    self.op = op # 产生该Node的Op
    self.grad = 0.0 # 初始化梯度
    self.evaluate() # 立即求值
    # 调试信息
    self.id = Node.global_id
    Node.global_id += 1
    print("eager exec: %s" % self)
  
  def input2values(self):
    """ 将输入统一转换成数值，因为具体的计算只能发生在数值上 """
    new_inputs = []
    for i in self.inputs:
      if isinstance(i, Node):
        i = i.value
      new_inputs.append(i)
    return new_inputs

  def evaluate(self):
    self.value = self.op.compute(self.input2values())

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return "Node%d: %s %s = %s, grad: %.3f" % (
      self.id, self.input2values(), self.op.name(), self.value, self.grad)

class Op(object):
  """
  所有操作的基类。注意Op本身不包含状态，计算的状态保存在Node中，每次调用Op都会产生一个Node。
  """
  def name(self):
    pass
  
  def __call__(self):
    """ 产生一个新的Node，表示此次计算的结果 """
    pass

  def compute(self, inputs):
    """ Op的计算 """
    pass

  def gradient(self, output_grad):
    """ 计算梯度 """
    pass

class AddOp(Op): 
  """加法运算"""
  def name(self):
    return "add"

  def __call__(self, a, b):
    return Node(self, [a, b])
  
  def compute(self, inputs):
    return inputs[0] + inputs[1]
  
  def gradient(self, inputs, output_grad):
    return [output_grad, output_grad] # gradient of a and b

class SubOp(Op): 
  """减法运算"""
  def name(self):
    return "sub"

  def __call__(self, a, b):
    return Node(self, [a, b])
  
  def compute(self, inputs):
    return inputs[0] - inputs[1]
  
  def gradient(self, inputs, output_grad):
    return [output_grad, -output_grad]

class MulOp(Op): 
  """乘法运算"""
  def name(self):
    return "mul"
  
  def __call__(self, a, b):
    return Node(self, [a, b])
  
  def compute(self, inputs):
    return inputs[0] * inputs[1]

  def gradient(self, inputs, output_grad):
    return [inputs[1] * output_grad, inputs[0] * output_grad]

class LnOp(Op): 
  """自然对数运算"""
  def name(self):
    return "ln"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return math.log(inputs[0])
  
  def gradient(self, inputs, output_grad):
    return [1.0/inputs[0] * output_grad]

class SinOp(Op): 
  """正弦运算"""
  def name(self):
    return "sin"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return math.sin(inputs[0])
  
  def gradient(self, inputs, output_grad):
    return [math.cos(inputs[0]) * output_grad]

class CosOp(Op): 
  """余弦运算"""
  def name(self):
    return "cos"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return math.cos(inputs[0])
  
  def gradient(self, inputs, output_grad):
    return [-math.sin(inputs[0]) * output_grad]

class TanOp(Op): 
  """正切运算"""
  def name(self):
    return "tan"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return math.tan(inputs[0])
  
  def gradient(self, inputs, output_grad):
    return [(1.0/(math.cos(inputs[0]))**2) * output_grad]

class ExpOp(Op): 
  """自然对数的指数运算"""
  def name(self):
    return "exp"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return math.exp(inputs[0])
  
  def gradient(self, inputs, output_grad):
    return [math.exp(inputs[0]) * output_grad]

class IdentityOp(Op): 
  """输入输出一样"""
  def name(self):
    return "identity"

  def __call__(self, a):
    return Node(self, [a])

  def compute(self, inputs):
    return inputs[0]
  
  def gradient(self, inputs, output_grad):
    return [output_grad]

class Executor(object):
  """ 计算图的执行和自动微分 """

  def __init__(self, root):
    self.topo_list = self.__topological_sorting(root) # 拓扑排序的顺序就是正向求值的顺序
    self.root = root

  def run(self):
    """
    按照拓扑排序的顺序对计算图求值。注意：因为我们之前对node采用了eager模式，
    实际上每个node值之前已经计算好了，但为了演示lazy计算的效果，这里使用拓扑
    排序又计算了一遍。
    """
    node_evaluated = set() # 保证每个node只被求值一次
    print("\nEVALUATE ORDER:")
    for n in self.topo_list:
      if n not in node_evaluated:
        n.evaluate()
        node_evaluated.add(n)
        print("evaluate: %s" % n)
    
    return self.root.value

  def __dfs(self, topo_list, node):
    if Node == None or not isinstance(node, Node):
      return
    for n in node.inputs:
      self.__dfs(topo_list, n)
    topo_list.append(node) # 同一个节点可以添加多次，他们的梯度会累加

  def __topological_sorting(self, root):
    """拓扑排序：采用DFS方式"""
    lst = []
    self.__dfs(lst, root)
    return lst

  def gradients(self):
    reverse_topo = list(reversed(self.topo_list)) # 按照拓扑排序的反向开始微分
    reverse_topo[0].grad = 1.0 # 输出节点梯度是1.0
    for n in reverse_topo:
      grad = n.op.gradient(n.input2values(), n.grad)
      # 将梯度累加到每一个输入变量的梯度上
      for i, g in zip(n.inputs, grad):
        if isinstance(i, Node):
          i.grad += g
    print("\nAFTER AUTODIFF:")
    for n in reverse_topo:
      print(n)


with open("AD.json") as f:
  data=json.load(f)

add, mul, ln, sin, sub, identity, cos, tan, exp = AddOp(), MulOp(), LnOp(), SinOp(), SubOp(), IdentityOp(), CosOp(), TanOp(), ExpOp()
x1, x2, x3 = identity(data['x1']), identity(data['x2']), identity(data['x3'])
y = eval(data['y'])
ex = Executor(y)
print("y=%.3f" % ex.run())
ex.gradients() # 反向计算 自动微分
print("x1.grad=%.3f" % x1.grad)
print("x2.grad=%.3f" % x2.grad)
print("x3.grad=%.3f" % x3.grad)

#利用数值微分进行验证：
def f(x1,x2,x3):
  return eval(data['z'])
e=0.0001
r1=(f(data['x1']+e,data['x2'],data['x3'])-f(data['x1'],data['x2'],data['x3']))/e
r2=(f(data['x1'],data['x2']+e,data['x3'])-f(data['x1'],data['x2'],data['x3']))/e
r3=(f(data['x1'],data['x2'],data['x3']+e)-f(data['x1'],data['x2'],data['x3']))/e
print("x1.grad from numerical differentiation=%.3f" % r1)
print("x2.grad from numerical differentiation=%.3f" % r2)
print("x3.grad from numerical differentiation=%.3f" % r3)