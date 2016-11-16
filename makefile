all: 
	make index.html

index.html: index.md
	pandoc -s --webtex -i -t slidy index.md -o index.html

clean:
	rm index.html

www:
	open http://data-index.github.io

open: index.html
	open index.html

edit:
	vi index.md

git: index.html
	git pull origin master;git commit -a ;git push origin master
