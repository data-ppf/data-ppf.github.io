all: 
	make index.html

index.html: ppf.md
	pandoc -s --webtex -i -t slidy ppf.md -o index.html

clean:
	rm index.html

www:
	open http://data-ppf.github.io

open: index.html
	open index.html

edit:
	vi ppf.md

git: index.html
	git pull origin master;git commit -a ;git push origin master
