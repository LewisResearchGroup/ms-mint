def button_style(style_name=None, **kwargs):
        style = {'color': 'black', 
                 'margin-right': 5, 
                 'margin-bottom': '1.5em',
                 'background-color': 'lightgrey'}
        
        styles = {'warn':    {'background-color': 'orange',
                              'font-weight': 'bold', 
                              'color': 'black'},
                  'error':   {'background-color': 'orange'},
                  'wait':    {'color': 'grey'},
                  'ready':   {'background-color': 'lightgreen',
                              'color': 'black'},
                  'help':    {'background-color': 'lightblue'},
                  'next':    {'font-weight': 'bold', 
                              'border': '2px solid blue'},
                  'neutral': {'background-color': 'white'}}
        
        if style_name in styles.keys():
            style.update(styles[style_name])
            
        style.update(**kwargs)

        return style
