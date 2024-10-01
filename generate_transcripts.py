import os

def summarize(fn):
    with open(fn) as f:
        text = f.read()
        
    summary = {}
    lines = iter(text.splitlines())
    grouped = []
    
    lines = text.splitlines()
    
    cur_chunk = {
        "is_voice": True,
        "content": []
    }
    for line in lines:
        if line.strip().isdigit():
            continue
        
        if line == "(Voice)" or line == "(User)":
            is_voice = line == "(Voice)"
            
            if cur_chunk["is_voice"] != is_voice:
                if cur_chunk is not None:
                    grouped.append((cur_chunk["is_voice"], ' '.join(cur_chunk["content"])))
                cur_chunk = {
                    "is_voice": is_voice,
                    "content": []
                }
        
            continue
        
        if "-->" in line:
            if 'I am now oriented' in line or 'Ik weet nu waar ik ben en wat ik moet doen' in line:
                cur_chunk["current_finish"] = line.split('--> ')[1]
            continue
        
        cur_chunk["content"].append(line)
        
    grouped.append((cur_chunk["is_voice"], ' '.join(cur_chunk["content"])))
        
    summary["interview"] = "unknown"
    summary["grouped"] = grouped
    
    return summary

dir = "transcripts"
durations = []
html = ""
for fn in os.listdir(dir):
    full_path = os.path.join(dir, fn)
    summary = summarize(full_path)
    
    try:
        durations.append(summary["interview"])
        html += "<div class='filename'>" + fn + "</div>\n"
        html += '<div class="duration">Interview duration: ' + summary["interview"] + "</div>\n"
        for is_voice, line in summary["grouped"]:
            class_name = "voice" if is_voice else "user"
            user_name = "Voice" if is_voice else "User"
            html += f'<div class="{class_name}">{user_name}: {line}</div>\n'
            
        print("good", fn)
    except KeyError:
        print("error", fn)
        pass
        
with open("transcripts.html", 'w') as f:
    f.write("""<html>
<style>
body {
    max-width: 40em;
    font-family: sans-serif;
}
.voice {
    background-color: #ffeeee;
}
div {
    margin: 1em;
}
.duration {
    text-align: right;
}
.filename {
    font-size: 2em;
    font-family: monospace;
    margin: 0;
    width: 100%;
    text-align: right;
    background-color: #ddffdd;
}
</style>
<body>""" + html + """</body>
</html>
""")
