all: 
	make index.html

index.html: ppf.md
	pandoc -s --webtex -i -t slidy ppf.md -o index.html

clean:
	rm index.html
