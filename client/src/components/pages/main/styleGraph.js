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
      content: (elem) => Number.parseFloat(elem.data('values').weight).toFixed(3),
      shape: 'rectangle',
      'background-color': function (elem) {
        let weight = elem.data('values').weight
        let maxWeight = elem.data('values').maxWeight
        // Меняем цвет данных в зависмости от данных.
        if (maxWeight > 0 && weight > 0) {
          let value = Math.min(255, parseInt((255 * weight) / maxWeight))
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
      content: (elem) => 'Data',
      shape: 'rectangle'
    }
  },

  {
    selector: '.linear',
    style: {
      content: (elem) => 'Linear\nbias:' + Number.parseFloat(elem.data('values').bias).toFixed(4),
      shape: 'round-rectangle'
    }
  },

  {
    selector: '.activation',
    style: {
      content: (elem) => 'Activation\ntype: ' + elem.data('values').actType,
      shape: 'cut-rectangle'
    }
  },

  {
    selector: '.maxPool2d',
    style: {
      content: (elem) =>
        'MaxPool2d' +
        '\nkernel size: ' +
        elem.data('values').kernelSize +
        '\npadding: ' +
        elem.data('values').padding +
        '\nstride: ' +
        elem.data('values').stride,
      shape: 'cut-rectangle'
    }
  },

  {
    selector: '.mergeFlatten',
    style: {
      content: (elem) => 'Merge Flatten',
      shape: 'cut-rectangle'
    }
  },

  {
    selector: '.reshape',
    style: {
      content: (elem) => 'Reshape',
      shape: 'cut-rectangle'
    }
  },

  {
    selector: '.convolution',
    style: {
      content: (elem) =>
        'Conv2d:' +
        '\npadding: ' +
        elem.data('values').padding +
        '\nstride: ' +
        elem.data('values').stride +
        '\nbias: ' +
        Number.parseFloat(elem.data('values').bias).toFixed(4),
      shape: 'cut-rectangle'
    }
  },

  {
    selector: '.model',
    style: {
      content: (elem) =>
        elem.data('values').name +
        '\nloss: ' +
        Number.parseFloat(elem.data('values').loss).toFixed(4),
      shape: 'cut-rectangle',
      'font-size': '50px',
      'background-opacity': 0,
      'z-index': -1,
      'border-width': '4',
      events: 'no'
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
      content: (elem) => Number.parseFloat(elem.data('values').weight).toFixed(6),
      'line-color': 'red',
      'line-opacity': 0.5,
      'z-index': 1
    }
  }
]
