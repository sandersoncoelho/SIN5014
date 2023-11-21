import math

from mathUtils import getAngle


class Polygon:
  def __init__(self, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    self.p1 = p1
    self.p2 = p2
    self.p3 = p3
    self.p4 = p4
    self.p5 = p5
    self.p6 = p6
    self.p7 = p7
    self.p8 = p8
    self.p9 = p9
    self.p10 = p10

    self.edge12 = math.dist(p1, p2)
    self.edge24 = math.dist(p2, p4)
    self.edge45 = math.dist(p4, p5)
    self.edge57 = math.dist(p5, p7)
    self.edge79 = math.dist(p7, p9)
    self.edge910 = math.dist(p9, p10)
    self.edge108 = math.dist(p10, p8)
    self.edge86 = math.dist(p8, p6)
    self.edge63 = math.dist(p6, p3)
    self.edge31 = math.dist(p3, p1)

    self.angle1 = getAngle(p3, p1, p2)
    self.angle2 = getAngle(p1, p2, p4)
    self.angle3 = getAngle(p2, p4, p5)
    self.angle4 = getAngle(p4, p5, p7)
    self.angle5 = getAngle(p5, p7, p9)
    self.angle6 = getAngle(p7, p9, p10)
    self.angle7 = getAngle(p9, p10, p8)
    self.angle8 = getAngle(p10, p8, p6)
    self.angle9 = getAngle(p8, p6, p3)
    self.angle10 = getAngle(p6, p3, p1)

  def __str__(self):
    return f"[{self.p1}, {self.p2}, {self.p3}, {self.p4}, {self.p5}, {self.p6}, {self.p7}, {self.p8}, {self.p9}, {self.p10}]"
  
class EdgeMeanStd:
  def __init__(self, edgeMean, edgeStd):
    self.edgeMean = edgeMean
    self.edgeStd = edgeStd

class EdgeStandart:
  def __init__(self, edge12, edge24, edge45, edge57, edge79, edge910, edge108, edge86, edge63, edge31):
    self.edge12 = edge12
    self.edge24 = edge24
    self.edge45 = edge45
    self.edge57 = edge57
    self.edge79 = edge79
    self.edge910 = edge910
    self.edge108 = edge108
    self.edge86 = edge86
    self.edge63 = edge63
    self.edge31 = edge31

class AngleMeanStd:
  def __init__(self, angleMean, angleStd):
    self.angleMean = angleMean
    self.angleStd = angleStd

class AngleStandart:
  def __init__(self, angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8, angle9, angle10):
    self.angle1 = angle1
    self.angle2 = angle2
    self.angle3 = angle3
    self.angle4 = angle4
    self.angle5 = angle5
    self.angle6 = angle6
    self.angle7 = angle7
    self.angle8 = angle8
    self.angle9 = angle9
    self.angle10 = angle10