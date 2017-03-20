#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
KD Tree
'''
import math
import functools
from bounded_priority_queue import BoundedPriorityQueue

class Node(object):
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
    
    def __repr__(self):
        return '<%(cls)s - %(data)s>' % dict(cls=self.__class__.__name__, data=repr(self.data))

    def preorder(self):
        if not self:
            return
        yield self
        if self.left:
            for x in self.left.preorder():
                yield x
        if self.right:
            for x in self.right.preorder():
                yield x
    
    def inorder(self):
        if not self:
            return
        if self.left:
            for x in self.left.inorder():
                yield x
        yield self
        if self.right:
            for x in self.right.inorder():
                yield x

    def postorder(self):
        if not self:
            return
        if self.left:
            for x in self.left.postorder():
                yield x
        if self.right:
            for x in self.right.postorder():
                yield x
        yield self


def require_axis(f):
    @functools.wraps(f)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) requires the node %(node)s '
                'to have an axis and a sel_axis function' %
                dict(func_name=f.__name__, node=repr(self)))
        return f(self, *args, **kwargs)
    return _wrapper


class KDNode(Node):
    """ A Node that contains kd-tree specific data and methods"""

    def __init__(self, data=None, left=None, right=None, axis=None, sel_axis=None, dimensions=None):
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions
    
    def axis_dist(self, point, axis):
        return math.pow(self.data[axis] - point[axis], 2)

    def _search_node(self, point, k, results, get_dist):
        if not self:
            return
        nodeDist = get_dist(self)
        results.add((self, nodeDist))
        split_plane = self.data[self.axis]
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist * plane_dist
        
        # Search the side of the spliting plane that the point is in
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist)
        
        # Search the other side of the splitting plane if it may contain
        # points closer than the farthest point in the current results.
        if plane_dist2 < results.max() or results.size() < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist)

    def dist(self, point):
        """
        Squared distance between the current Node and the given point
        """
        r = range(self.dimensions)
        return sum([self.axis_dist(point, i) for i in r])

    def search_knn(self, point, k, dist=None):
        """
        k is the number of results to return. The actual results can be less
        (if there aren't more nodes to return) or more in case of equal distances.
        dist is a distance function, expecting two points and returning a distance value.
        The result is an ordered list of (node, distance) tuples.
        """
        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            get_dist = lambda n: dist(n.data, point)
        
        results = BoundedPriorityQueue(k)
        self._search_node(point, k, results, get_dist)

        BY_VALUE = lambda kv: kv[1]
        return sorted(results.items(), key=BY_VALUE)

    @require_axis
    def search_nn(self, point, dist=None):
        """
        search the nearest node of the given point
        """
        return next(iter(self.search_knn(point, 1, dist)), None)


def check_dimensions(point_list, dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list[1:]:
        if len(p) != dimensions:
            raise ValueError('All points in must have the same dimension')
    return dimensions
    

def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    """
    axis is the axis on which current node should split.
    sel_axis(axis) is used when creating subnodes of a node.
    It receives the axis of the parent node and returns the axis of the child node.
    """
    if not point_list and not dimensions:
        raise ValueError('Either point_list or dimensions must be provided.')
    elif point_list:
        dimensions = check_dimensions(point_list, dimensions)
    
    # by default cycle through the axis
    sel_axis = sel_axis or (lambda pre_axis: (pre_axis + 1) % dimensions)

    if not point_list:
        return None
        # return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)
    
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc = point_list[median]
    left = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median + 1:], dimensions, sel_axis(axis))
    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


if __name__ == '__main__':
    points_list = [(2,3),[5,4],[9,6],[4,7],[8,1],[7,2]]
    tree = create(points_list)
    print list(tree.preorder())
    print tree.search_nn([6,10])
    print tree.search_knn([6,10], 2)