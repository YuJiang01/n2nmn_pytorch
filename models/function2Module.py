
from models.modules import *

function2module = {
    'filter_color': FilterModule,
    'filter_material': FilterModule,
    'filter_shape': FilterModule,
    'filter_size': FilterModule,

    'same_color': FindSamePropertyModule,
    'same_material': FindSamePropertyModule,
    'same_shape': FindSamePropertyModule,
    'same_size': FindSamePropertyModule,

    'relate': TransformModule,
    'intersect': AndModule,
    'union': OrModule,

    'count': CountModule,
    'exist': ExistModule,
    'equal_integer': EqualNumModule,
    'greater_than': MoreNumModule,
    'less_than': LessNumModule,

    'equal_color': SamePropertyModule,
    'equal_material': SamePropertyModule,
    'equal_shape': SamePropertyModule,
    'equal_size': SamePropertyModule,

    'query_color': DescribeModule,
    'query_material': DescribeModule,
    'query_shape': DescribeModule,
    'query_size': DescribeModule,

    'scene': SceneModule,
    'unique': None
}