/**
 * Texts for assistive technology (docs/accessibility.md): the skip link, the labels and
 * descriptions of the canvas, what the live region announces, and the keyboard way of
 * connecting layers in the properties panel.
 */
export const a11y = {
  skipToContent: 'Skip to content',
  mainNav: 'Main',
  canvas: {
    label: 'Network canvas',
    nodeDescription:
      'Layer on the canvas. Enter or Space selects it; arrow keys then move it (Shift: faster), Delete removes it, Escape clears the selection. Its settings and connections are in the properties panel.',
    nodeDescriptionReadOnly: 'Layer on the canvas. Enter or Space selects it to show its settings.',
    edgeDescription:
      'Connection on the canvas. Enter or Space selects it; Delete then removes it, Escape clears the selection.',
    moved: 'Moved {{direction}}. Position: x {{x}}, y {{y}}.',
    directions: { left: 'left', right: 'right', up: 'up', down: 'down' },
    controls: 'Canvas controls',
    zoomIn: 'Zoom in',
    zoomOut: 'Zoom out',
    fitView: 'Fit the network into view',
    interactive: 'Lock or unlock moving layers',
    minimap: 'Overview of the network',
    handle: 'Connection point',
    node: '{{name}}, {{type}}',
    nodeWithErrors: '{{name}}, {{type}}, has problems',
    edge: 'Connection from {{source}} to {{target}}',
  },
  /** Announced (politely) when the network changes by an action of the user. */
  live: {
    added: 'Added {{name}}.',
    addedAfter: 'Added {{name}} after {{after}}.',
    addedInto: 'Added {{name}}, connected to {{into}}.',
    connected: 'Connected {{source}} to {{target}}.',
    disconnected: 'Removed the connection from {{source}} to {{target}}.',
    deleted: 'Deleted {{name}}.',
    deletedMany_one: 'Deleted {{count}} layer.',
    deletedMany_other: 'Deleted {{count}} layers.',
    selected: 'Selected {{name}}.',
  },
  /** The connections of the selected layer in the properties panel. */
  connections: {
    title: 'Connections',
    inputs: 'Inputs from',
    outputs: 'Outputs to',
    none: 'None',
    addInput: 'Add input from …',
    addOutput: 'Add output to …',
    addInputHint: 'Layers whose output this layer can take.',
    addOutputHint: 'Layers that can take the output of this layer.',
    noCandidates: 'No other layer fits here.',
    remove: 'Remove the connection from {{source}} to {{target}}',
    select: 'Select {{name}}',
  },
  /** Model summary: the layers as a list, the text alternative to the canvas. */
  layers: {
    title: 'Layers',
    label: 'Layers of the network, in the order they were added',
    inputsFrom: 'input from {{names}}',
    noInputs: 'no inputs',
  },
  /** The window title of each page. */
  titles: {
    format: '{{page}} · Brick DNN',
    authorize: 'Connect an app',
    shared: 'Shared network',
    comparison: 'Shared comparison',
  },
  charts: {
    noValue: 'no value',
    epoch: 'Epoch',
  },
};
