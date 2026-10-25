"""Catalog official Afghan curriculum PDF links without downloading the books."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

SOURCES = {
    "pashto": "https://www.moe.gov.af/index.php/en/node/450",
    "dari": "https://www.moe.gov.af/dr/كتب-نصاب-تعليمى",
}
DEST = Path("data/raw/afghan_curriculum")


def safe_url(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme,p.netloc,urllib.parse.quote(p.path,safe="/%"),
                                   urllib.parse.quote(p.query,safe="=&%"),p.fragment))


class CatalogParser(HTMLParser):
    def __init__(self, source, language):
        super().__init__(convert_charrefs=True)
        self.source,self.language=source,language
        self.records=[]
        self.table=-1;self.depth=0;self.row=-1;self.column=0
        self.cell=None;self.anchor=None;self.header=[];self.grade_map={}

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='table':
            self.depth+=1
            if self.depth==1:
                self.table+=1;self.row=-1;self.grade_map={}
        if self.depth!=1:return
        if tag=='tr':self.row+=1;self.column=0;self.header=[]
        if tag in ('td','th'):
            self.cell={'text':[], 'links':[], 'span':int(attrs.get('colspan','1'))}
        if tag=='a' and self.cell is not None:
            self.anchor={'href':attrs.get('href',''),'text':[]}

    def handle_data(self,data):
        if self.cell is not None:self.cell['text'].append(data)
        if self.anchor is not None:self.anchor['text'].append(data)

    def handle_endtag(self,tag):
        if tag=='a' and self.anchor is not None and self.cell is not None:
            self.cell['links'].append(self.anchor);self.anchor=None
        if tag in ('td','th') and self.cell is not None:
            cell=self.cell
            label=' '.join(''.join(cell['text']).split())
            if self.row==0:self.header.append((self.column,cell['span'],label))
            grade=self.grade_map.get(self.column)
            # During the first row no PDFs are expected; it defines the columns.
            for a in cell['links']:
                url=urllib.parse.urljoin(self.source,a['href'])
                if not urllib.parse.urlsplit(url).path.lower().endswith('.pdf'):continue
                title=' '.join(''.join(a['text']).split()) or label
                filename=urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)
                match=re.search(r'(?i)\bG(\d{1,2})[-_]',filename)
                hint='dari' if re.search(r'(?i)(?:^|[-_])dr(?:[-_]|$)',filename) else 'pashto' if re.search(r'(?i)(?:^|[-_])ps(?:[-_]|$)',filename) else 'unknown'
                self.records.append({'source_page':self.source,'catalog_language':self.language,
                    'table':self.table,'row':self.row,'column':self.column,
                    'grade':int(match[1]) if match else grade,'title':title,
                    'medium_hint':hint,'content_language_verified':False,
                    'url':safe_url(url),'original_filename':filename})
            self.column+=cell['span'];self.cell=None
        if tag=='tr' and self.row==0:
            start=7 if self.header and any(word in self.header[0][2] for word in ('اووم','هفتم')) else 1
            if self.header and any(word in self.header[0][2] for word in ('ټولګ','صنف')):
                for i,(column,span,label) in enumerate(self.header):
                    for c in range(column,column+span):self.grade_map[c]=start+i
        if tag=='table':self.depth-=1


def main():
    if (DEST/'catalog.json').exists():
        raise SystemExit('Catalog already exists; preserving it.')
    records=[];sources=[]
    for language,url in SOURCES.items():
        print('Reading official catalog:',language,flush=True)
        with urllib.request.urlopen(safe_url(url),timeout=40) as response:
            raw=response.read(4*1024*1024+1)
        if len(raw)>4*1024*1024:raise ValueError('Catalog exceeds size limit')
        html=raw.decode('utf-8')
        parser=CatalogParser(url,language);parser.feed(html)
        if not parser.records:raise ValueError(f'No PDF links found for {language}')
        DEST.mkdir(parents=True,exist_ok=True)
        (DEST/f'{language}_source.html').write_bytes(raw)
        sources.append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'pdf_references':len(parser.records)})
        records.extend(parser.records)
    unique={}
    for row in records:
        key=row['url']
        if key not in unique:
            row['id']=hashlib.sha256(key.encode()).hexdigest()[:12]
            row['aliases']=[]
            unique[key]=row
        else:
            unique[key]['aliases'].append({k:row[k] for k in ('catalog_language','grade','title')})
    report={'sources':sources,'unique_pdf_urls':len(unique),'books':list(unique.values()),
            'rights_status':'Public access observed; training/republication permission not verified.',
            'notes':'Grade and medium are catalog/filename hints; verify against each PDF. No books downloaded by this command.'}
    with (DEST/'catalog.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print('Saved:',DEST/'catalog.json','Unique PDF URLs:',len(unique))
    for language in SOURCES:
        examples=[b for b in unique.values() if b['catalog_language']==language and b['grade']==1]
        print(language,'grade 1 candidates:',[(b['id'],b['title'],b['original_filename']) for b in examples[:12]])


if __name__=='__main__':main()
