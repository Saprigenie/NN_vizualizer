export let graphStyles = [
  // the stylesheet for the graph
  {
    selector: 'node',
    style: {
      width: (elem) => elem.data('width') + 'px',
      height: (elem) => elem.data('height') + 'px',
      'text-wrap': 'wrap',
      'background-color': '#666'
    }
  },

  {
    selector: '.textContrast',
    style: {
      color: 'black',
      'text-outline-width': '1',
      'text-outline-color': 'white'
    }
  },

  {
    selector: '.textWhite',
    style: {
      color: 'white'
    }
  },

  {
    selector: '.textTop',
    style: {
      'text-valign': 'top'
    }
  },

  {
    selector: '.textCenter',
    style: {
      'text-valign': 'center',
      'text-halign': 'center'
    }
  },

  {
    selector: '.border',
    style: {
      'border-color': '#666',
      'border-width': '2'
    }
  },

  {
    selector: '.blackBorder',
    style: {
      'border-color': 'black',
      'border-width': '2'
    }
  },

  {
    selector: '.data',
    style: {
      content: (elem) => elem.data('value'),
      shape: 'rectangle',
      'background-color': function (elem) {
        // Меняем цвет данных в зависмости от данных.
        if (elem.data('value') > 0) {
          let value = Math.min(255, parseInt((255 * elem.data('value')) / 16))
          return 'rgb(' + value + ',' + value + ',' + value + ')'
        } else {
          return 'black'
        }
      }
    }
  },

  {
    selector: '.dataImage',
    style: {
      content: (elem) => elem.data('value'),
      shape: 'rectangle'
    }
  },

  {
    selector: '.linear',
    style: {
      content: (elem) => elem.data('value'),
      shape: 'round-rectangle'
    }
  },

  {
    selector: '.activation',
    style: {
      content: (elem) => elem.data('value'),
      shape: 'cut-rectangle'
    }
  },

  {
    selector: '.convolution',
    style: {
      content: (elem) => elem.data('value'),
      shape: 'cut-rectangle'
    }
  },

  {
    selector: 'edge',
    style: {
      width: 3,
      'text-outline-width': '1',
      'text-outline-color': 'white',
      'text-outline-opacity': '1'
    }
  },

  {
    selector: '.ehasweights',
    style: {
      'line-color': 'black',
      opacity: '0.1'
    }
  },

  {
    selector: '.enothasweights',
    style: {
      'line-color': '#ccc'
    }
  },

  {
    selector: '.highlight',
    style: {
      'line-color': 'green',
      'border-color': 'green',
      'border-width': '10px'
    }
  },

  {
    selector: '.edisplayweights',
    style: {
      content: 'data(value)',
      'line-color': 'red',
      'line-opacity': 0.5,
      'z-index': 1
    }
  }
]
