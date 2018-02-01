
from models.modules import *

layout2module = {
    '_Filter': FilterModule,
    '_FindSameProperty': FindSamePropertyModule,
    '_Transform': TransformModule,
    '_And': AndModule,
    '_Or': OrModule,

    '_Count': CountModule,
    '_Exist': ExistModule,
    '_EqualNum': EqualNumModule,
    '_MoreNum': MoreNumModule,
    '_LessNum': LessNumModule,

    '_SameProperty': SamePropertyModule,

    '_Describe': DescribeModule,


    '_Scene': SceneModule,
    '_Find': FindModule
}