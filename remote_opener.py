import tempfile
import os
import webbrowser


def openBrowser(open_url):
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "remote_temp.html")

    # to open/create a new html file in the write mode
    f = open(save_path, 'w')

    # the html code which will go in the file GFG.html
    html_template = """<html>
    <head>
    <title>Remote Control Opener</title>
        <meta http-equiv="Refresh" content="0; url='""" + open_url + """'" />
    </head>
    <body>
    <h2>Remote Control Opener</h2>
    
        <p><a href='""" + open_url + """' target='_self'>Open Websocket Remote Control Webapp if not opened automatically by clicking here.</a></p>
    
    </body>
    </html>
    """

    # writing the code into the file
    f.write(html_template)

    # close the file
    f.close()

    webbrowser.open('file://' + save_path, new=1)
