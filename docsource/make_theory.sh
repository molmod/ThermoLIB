fn=/home/louis/Documents/Papers/ThermoLIB/TheoryDraft/main.tex
cp $fn theory.tex
iconv -t utf-8 theory.tex | pandoc -f latex -t rst -o theory-source.rst | iconv -f utf-8