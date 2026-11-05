"""Read the real visible source counter from captured Blender screen frames.

Optionally select one recorded frame per source frame for a 30 fps release
master. No scene rendering, motion interpolation or synthesized UI is involved.
"""
import argparse
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path)
    p.add_argument('--start',type=float,default=0)
    p.add_argument('--duration',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select',type=Path)
    p.add_argument('--frames',type=int,default=750)
    a=p.parse_args()
    cv2.setNumThreads(1)
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=avg_frame_rate','-of','json',str(a.video)],text=True))
    numerator,denominator=map(int,metadata['streams'][0]['avg_frame_rate'].split('/'))
    fps=numerator/denominator;first=round(a.start*fps)
    decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-ss',str(a.start),'-i',str(a.video),
        '-t',str(a.duration),'-an','-filter_complex',
        '[0:v]split=2[a][b];[a]crop=40:17:1440:735,pad=40:21:0:0:black[a1];'
        '[b]crop=91:21:219:75[b1];[a1][b1]hstack,format=gray[out]',
        '-map','[out]','-f','rawvideo','pipe:1'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def glyphs(roi):
        _,_,stats,_=cv2.connectedComponentsWithStats(roi)
        result=[]
        for x,y,w,h,area in sorted(stats[1:].tolist(),key=lambda v:v[0]):
            if area<6 or w>10 or h>13:continue
            value=np.zeros((13,10),np.uint8);value[:h,:w]=roi[y:y+h,x:x+w];result.append(value)
        return result
    rows=[]
    for index in range(round(a.duration*fps)):
        data=decoder.stdout.read(131*21)
        if len(data)!=131*21:break
        image=(np.frombuffer(data,np.uint8).reshape(21,131)>150).astype(np.uint8)
        source=glyphs(image[:17,:40]);stage=glyphs(image[:,40:])
        # The slash is taller than digits and filtered; discard denominator.
        stage=stage[:-3] if len(stage)>=4 else []
        rows.append((first+index,source,stage))
    decoder.stdout.close()
    if decoder.wait(timeout=30):raise RuntimeError(decoder.stderr.read().decode())
    def learn(column):
        one=[]
        for row in rows:
            values=row[column]
            if len(values)==1:
                value=values[0]
                if not one or min(np.count_nonzero(value!=x) for x in one)>4:one.append(value)
                if len(one)>9:raise AssertionError(('Unexpected single-digit glyphs',column))
            elif len(values)==2 and len(one)==9:
                templates={str(i+1):value for i,value in enumerate(one)};templates['0']=values[1];return templates
        raise AssertionError(('Could not learn counters 1 through 10',column))
    source_templates,stage_templates=learn(1),learn(2)
    maximum=[0,0]
    def decode(values,templates,column):
        if not values:return None
        digits=[]
        for value in values:
            error,digit=min((min(np.count_nonzero(value!=np.roll(template,(dy,dx),(0,1)))
                for dy in (-1,0,1) for dx in (-1,0,1)),digit) for digit,template in templates.items())
            if error>16:raise AssertionError((column,int(error),digit))
            maximum[column]=max(maximum[column],int(error));digits.append(digit)
        return int(''.join(digits))
    sequences=[[],[]];occurrences={};selected={}
    for frame,source,stage in rows:
        numbers=[decode(source,source_templates,0),decode(stage,stage_templates,1)]
        for column,number in enumerate(numbers):
            if number is not None and (not sequences[column] or sequences[column][-1]!=number):sequences[column].append(number)
        if numbers[0] is not None:occurrences[numbers[0]]=occurrences.get(numbers[0],0)+1
        if numbers[0] is not None and numbers[0]==numbers[1]:selected.setdefault(numbers[0],[]).append(frame)
    for column,sequence in enumerate(sequences):
        if sequence!=list(range(1,a.frames+1)):raise AssertionError((column,len(sequence),sequence[:12],sequence[-12:]))
    missing=set(range(1,a.frames+1))-selected.keys()
    if missing:raise AssertionError(('No encoded frame with matching source and stage counter',sorted(missing)))
    mapping={n:indices[len(indices)//2] for n,indices in selected.items()}
    report={'video':str(a.video.resolve()),'recording_fps':fps,'source_frames_seen':a.frames,
        f'exact_sequence_1_through_{a.frames}':True,'stage_source_counters_match_for_every_selected_frame':True,
        'encoded_frames_inspected':len(rows),'minimum_captures_per_source_frame':min(occurrences.values()),
        'max_glyph_pixel_mismatch':maximum,'selected_recording_frame_per_source_frame':mapping,
        'method':'Read both visible counters from actual screen frames; select a recorded frame with matching stage/source for each original frame.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.select:
        a.select.parent.mkdir(parents=True,exist_ok=True)
        first_index=min(mapping.values());last_index=max(mapping.values())
        choices='+'.join(f'eq(n,{index-first_index})' for index in mapping.values())
        subprocess.run(['ffmpeg','-v','error','-threads','2','-ss',str(first_index/fps),'-i',str(a.video),
            '-t',str((last_index-first_index+1)/fps+.01),'-an',
            '-vf',f"select='{choices}',setpts=N/(30*TB)",'-r','30','-c:v','libx264',
            '-threads','4','-preset','fast','-crf','16','-pix_fmt','yuv420p',
            '-movflags','+faststart','-y',str(a.select)],check=True)
        report['selected_video']=str(a.select.resolve());report['selected_duration']=a.frames/30
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='selected_recording_frame_per_source_frame'}))


if __name__=='__main__':main()
