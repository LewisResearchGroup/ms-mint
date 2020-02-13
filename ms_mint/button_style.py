def button_style(style_name=None, **kwargs):
        style = {'color': 'black', 
                 'margin-right': 20, 
                 'margin-bottom': '1.5em',
                 'background-color': 'lightgrey'}
        
        styles = {'warn': {'font-weight': 'bold', 'color': 'red'},
                  'error': {'background-color': 'orange'},
                  'wait': {'color': 'grey'},
                  'ready': {'background-color': 'lightgreen',
                            'color': 'black'},
                  'help': {'background-color': 'lightblue'},
                  'next': {'font-weight': 'bold', 
                           'border': '2px solid blue'}}
        
        if style_name in styles.keys():
            style.update(styles[style_name])
            
        style.update(**kwargs)

        return style
