
from models.modules import *

layout2module = {
    '_Filter': self.FilterModule,
    '_FindSameProperty': self.FindSamePropertyModule,
    '_Transform': self.TransformModule,
    '_And': self.AndModule,
    '_Or': self.OrModule,

    '_Count': self.CountModule,
    '_Exist': self.ExistModule,
    '_EqualNum': self.EqualNumModule,
    '_MoreNum': self.MoreNumModule,
    '_LessNum': self.LessNumModule,

    '_SameProperty': self.SamePropertyModule,

    '_Describe': self.DescribeModule,


    '_Scene': self.SceneModule,
    '_Find': self.FindModule
}