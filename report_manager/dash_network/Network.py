# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class Network(Component):
    """A Network component.
Network graph component, based on D3 force layout

Keyword arguments:
- id (string; optional): The ID used to identify this component in Dash callbacks
- width (number; optional): Width of the figure to draw, in pixels
- height (number; optional): Height of the figure to draw, in pixels
- data (dict; required): The network data. Should have the form:

  `{nodes: [node0, node1, ...], links: [link0, link1, ...]}`

nodes have the form:

  `{id: 'node id'[, radius: number][, color: 'css color string']}`

`id` is required, must be unique, and is used both in links and
as the node text.
`radius` is an optional relative radius, scaled by `maxRadius`
`color` is an optional css color string.

links have the form:

  `{source: sourceId, target: targetId[, width: number]}`

`source` and `target` are required, and must match node ids.
`width` is an optional relative width, scaled by `maxLinkWidth`
- dataVersion (string | number; optional): Optional version id for data, to avoid having to diff a large object
- linkWidth (number; optional): Optional default width of links, in px
- maxLinkWidth (number; optional): Optional maximum width of links, in px. If individual links have `width`,
these will be scaled linearly so the largest one has width `maxLinkWidth`.
- nodeRadius (number; optional): Optional default radius of nodes, in px
- maxRadius (number; optional): Optional maximum radius of nodes, in px. If individual nodes have `radius`,
these will be scaled linearly so the largest one has radius `maxRadius`.
- selectedId (string; optional): The currently selected node id

Available events: """
    @_explicitize_args
    def __init__(self, id=Component.UNDEFINED, width=Component.UNDEFINED, height=Component.UNDEFINED, data=Component.REQUIRED, dataVersion=Component.UNDEFINED, linkWidth=Component.UNDEFINED, maxLinkWidth=Component.UNDEFINED, nodeRadius=Component.UNDEFINED, maxRadius=Component.UNDEFINED, selectedId=Component.UNDEFINED, **kwargs):
        self._prop_names = ['id', 'width', 'height', 'data', 'dataVersion', 'linkWidth', 'maxLinkWidth', 'nodeRadius', 'maxRadius', 'selectedId']
        self._type = 'Network'
        self._namespace = 'dash_network'
        self._valid_wildcard_attributes =            []
        self.available_events = []
        self.available_properties = ['id', 'width', 'height', 'data', 'dataVersion', 'linkWidth', 'maxLinkWidth', 'nodeRadius', 'maxRadius', 'selectedId']
        self.available_wildcard_properties =            []

        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs
        args = {k: _locals[k] for k in _explicit_args if k != 'children'}

        for k in ['data']:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')
        super(Network, self).__init__(**args)

    def __repr__(self):
        if(any(getattr(self, c, None) is not None
               for c in self._prop_names
               if c is not self._prop_names[0])
           or any(getattr(self, c, None) is not None
                  for c in self.__dict__.keys()
                  if any(c.startswith(wc_attr)
                  for wc_attr in self._valid_wildcard_attributes))):
            props_string = ', '.join([c+'='+repr(getattr(self, c, None))
                                      for c in self._prop_names
                                      if getattr(self, c, None) is not None])
            wilds_string = ', '.join([c+'='+repr(getattr(self, c, None))
                                      for c in self.__dict__.keys()
                                      if any([c.startswith(wc_attr)
                                      for wc_attr in
                                      self._valid_wildcard_attributes])])
            return ('Network(' + props_string +
                   (', ' + wilds_string if wilds_string != '' else '') + ')')
        else:
            return (
                'Network(' +
                repr(getattr(self, self._prop_names[0], None)) + ')')
