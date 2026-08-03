%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Klasa do tworzenia prac dyplomowych 
% na Wydziale Matematyki PWr.
%
% Autor: Damian Fafuła
%Data utworzenia: 8.05.2018
%
% Zmodyfikowane przez: Jakuba Ślęzaka 
% Data utworzenia aktualnej wersji: 28.01.2025 
% Numer wersji: 1.1
%
% Poniższą klasę można rozpowszechniać i edytować 
% pod warunkiem zachowania numeru wersji, 
% informacji o autorach i dodaniu informacji 
% o wprowadzonych zmianach.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pwr_wmat_praca_dyplomowa}[2025/01/28 Praca dyplomowa]

\ifx\rozdzialyOdNieparzystych\undefined
	\PassOptionsToClass{12pt,a4paper,twoside}{book}
\else
	\PassOptionsToClass{12pt,a4paper,twoside,openany}{book}
\fi
\LoadClass{book}

\RequirePackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm,headheight=50pt]{geometry}
\RequirePackage{microtype}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{scrextend}
\let\lll\undefined
\RequirePackage{amsmath,amsfonts,amssymb,amsthm}
\RequirePackage{enumerate}
\RequirePackage{caption}
\RequirePackage{afterpage}
\RequirePackage{multirow}
\RequirePackage{enumitem}
\RequirePackage{array}
\RequirePackage[svgnames]{xcolor}
\RequirePackage{float}
\RequirePackage{xstring,xifthen}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}
\RequirePackage{stackengine} 
\RequirePackage{tikz}
\RequirePackage{arydshln}

\definecolor{col1}{RGB}{35,31,32}
\definecolor{col2}{RGB}{255,255,255}

\newcommand{\autor}[1]{\gdef\@autor{#1}}
\newcommand{\tytul}[1]{\gdef\@tytul{#1}}
\newcommand{\opiekun}[1]{\gdef\@opiekun{#1}}
\newcommand{\kierunekstudiow}[1]{\gdef\@kierunekstudiow{#1}}
\newcommand{\specjalnosc}[1]{\gdef\@specjalnosc{#1}}
\newcommand{\rodzajpracy}[1]{\gdef\@rodzajpracy{#1}}
\newcommand{\streszczenie}[1]{\gdef\@streszczenie{#1}}
\newcommand{\slowakluczowe}[1]{\gdef\@slowakluczowe{#1}}
\newcommand{\tytulang}[1]{\gdef\@tytulang{#1}}
\newcommand{\kierunekstudiowang}[1]{\gdef\@kierunekstudiowang{#1}}
\newcommand{\specjalnoscang}[1]{\gdef\@specjalnoscang{#1}}
\newcommand{\rodzajpracyang}[1]{\gdef\@rodzajpracyang{#1}}
\newcommand{\streszczenieang}[1]{\gdef\@streszczenieang{#1}}
\newcommand{\slowakluczoweang}[1]{\gdef\@slowakluczoweang{#1}}
\newlength{\szerokosc}

\ifx\koloroweLinki\undefined
	\RequirePackage[hidelinks]{hyperref}
\else
	\RequirePackage{hyperref}
	\newcommand\myshade{85}
	\colorlet{mylinkcolor}{DarkBlue}
	\colorlet{mycitecolor}{FireBrick}
	\colorlet{myurlcolor}{DarkGreen}
	\hypersetup{
		linkcolor  = mylinkcolor!\myshade!black,
		citecolor  = mycitecolor!\myshade!black,
		urlcolor   = myurlcolor!\myshade!black,
		colorlinks = true,
	}
\fi

\input{"logoPwr.tex"}

\def\dotfill#1{\cleaders\hbox to #1{.}\hfill}
\newcommand\dotline[2][.5em]{\leavevmode\hbox to #2{\dotfill{#1}\hfil}}
   
\def\dlugoscmaxa{450}  
\def\dlugoscmaxb{500}
\def\dlugoscmaxc{150}
\def\dlugoscmaxd{29}
\def\dlugoscmaxe{32}
\def\dlugoscmaxf{36} 
\def\dlugoscmaxg{80}
  
\newcommand{\stronatytulowapl}{
\tikzset{wydzial/.style={font=\bfseries}}
\tikzset{nazwa/.style={font=\rmfamily}}
\tikzset{tytul/.style={font=\scshape}}
\thispagestyle{empty}
\afterpage{
\newgeometry{left=4.8cm, right=3cm, top=3cm, bottom=3cm}
\begin{center}
	\begin{tikzpicture}[remember picture,overlay]
		\node[inner sep=0pt] at (-4.28,-.05){\logopl};
		\draw[white, fill=gray!25] (-8.77, -1) -- (-8.77,-23.5) -- (-6.84,-23.5) -- (-6.84,-1);
		\draw[-, col1,line width=.33mm] (-6.855,-.9885) -- (6.65,-.9885);
	\end{tikzpicture}
\end{center}
\hspace{1cm}
\vspace{-1.7ex}
\begin{flushleft}
\textbf{Wydział Matematyki} 

\vspace{0.15ex}
Kierunek studiów: \@kierunekstudiow

Specjalność: \@specjalnosc\\
\text{}
\end{flushleft}
\begin{minipage}[t][0.46\textheight]{\textwidth}
\begin{center}
	\large Praca dyplomowa --  \@rodzajpracy \\
	\vspace{.6cm plus .5fill}
	\StrLen{\@tytul}[\tytuldlugosc]
	\ifthenelse{\tytuldlugosc > \dlugoscmaxg}{
		\large
	    \textbf{\MakeUppercase{\@tytul}}\\
	}{
		\Large\textbf{\MakeUppercase{\@tytul}}\\
	}
	\vspace{0.6cm plus .2fill}
	\large \@autor
\end{center}
\vspace{0.1cm plus .2fill}
\begin{flushright}
\small
słowa kluczowe:\\
\vspace{0.5ex}
\begin{minipage}{7cm}
	\StrLen{\@slowakluczowe}[\slowakluczowedlugosc]
	\ifthenelse{\slowakluczowedlugosc > \dlugoscmaxc}{
		\footnotesize 
	    \@slowakluczowe
	}{
		\@slowakluczowe
	}
\end{minipage}
\end{flushright}
\vspace{0.07cm plus 0.2fill}
\begin{flushleft}
	\small krótkie streszczenie:\\[.8ex]
	\StrLen{\@streszczenie}[\streszczeniedlugosc]
	\ifthenelse{\streszczeniedlugosc > \dlugoscmaxb}{
		\parbox{\textwidth}{
	    	\KOMAoptions{fontsize=9pt}
	    	\large\@streszczenie
		}
	}{
		\ifthenelse{\streszczeniedlugosc > \dlugoscmaxa}{
			\parbox{\textwidth}{
			\@streszczenie
			}
		}{
			\parbox{\textwidth}{
			\footnotesize
			\@streszczenie
			}
		}
	}
	\end{flushleft}
\end{minipage}

\vspace{\fill}


\begin{minipage}[b][0.33\textheight]{\textwidth}
\flushright
\begin{tabular}{l}
 Promotor:\\
	\@opiekun
\end{tabular}
	
\vspace{0.8ex}
\emph{
\footnotesize
\begin{flushleft}
	Do celów archiwalnych pracę dyplomową zakwalifikowano do:*
	\vspace{-1ex}
	\begin{enumerate}
		\itemsep0em 
	    \item[a)] kategorii A (akta wieczyste)
	    \item[b)] kategorii BE 50 (po 50 latach podlegające ekspertyzie)
	\end{enumerate}
	\vspace{-1ex}
	* niepotrzebne skreślić
\end{flushleft}
}
\begin{flushright}
	\boxed{~~~\textrm{pieczątka wydziałowa}~~~}
\end{flushright}
\vspace{0cm}
\begin{center}
	\large Wrocław, rok \the\year
\end{center}
\end{minipage}
%\end{flushleft}
\clearpage
\restoregeometry
}
}

\newcommand{\stronatytulowaang}{
\tikzset{wydzial/.style={font=\bfseries}}
\tikzset{nazwa/.style={font=\rmfamily}}
\tikzset{tytul/.style={font=\scshape}}
\thispagestyle{empty}
\afterpage{
\newgeometry{left=4.8cm, right=3cm, top=3cm, bottom=3cm}
\begin{center}
\begin{tikzpicture}[remember picture,overlay]
\node[inner sep=0pt] at (-4.28,-.05){\logoang};
\draw[white, fill=gray!25] (-8.77, -1) -- (-8.77,-23.5) -- (-6.84,-23.5) -- (-6.84,-1);
\draw[-, col1,line width=.33mm] (-6.855,-.9885) -- (6.65,-.9885);
\end{tikzpicture}
\end{center}
\hspace{1cm}
\vspace{-1.7ex}
\begin{flushleft}
\textbf{Faculty of Pure and Applied Mathematics} 

\vspace{0.15ex}
Field of study: \@kierunekstudiowang

Specialty: \@specjalnoscang
\text{}
\end{flushleft}
\begin{minipage}[t][0.46\textheight]{\textwidth}
\begin{center}
\text{}

\large \@rodzajpracyang \\
\vspace{.6cm plus .5fill}
\StrLen{\@tytulang}[\tytulangdlugosc]
\ifthenelse{\tytulangdlugosc > \dlugoscmaxg}{
	\large
    \textbf{\MakeUppercase{\@tytulang}}\\
}{
	\Large\textbf{\MakeUppercase{\@tytulang}}\\
}
\vspace{0.6cm plus .2fill}
\large \@autor
\end{center}
\vspace{0.1cm plus .2fill}
\begin{flushright}
\small
keywords:\\
\vspace{1ex}
\begin{minipage}{7cm}
\StrLen{\@slowakluczoweang}[\slowakluczoweangdlugosc]
\ifthenelse{\slowakluczoweangdlugosc > \dlugoscmaxc}{
	\footnotesize 
    \@slowakluczoweang
}{
	\@slowakluczoweang
}
\end{minipage}
\end{flushright}
\vspace{0.07cm plus 0.2fill}
\begin{flushleft}
\small short summary:\\[.8ex]
\StrLen{\@streszczenieang}[\streszczenieangdlugosc]
\ifthenelse{\streszczenieangdlugosc > \dlugoscmaxb}{
	\parbox{\textwidth}{
    	\KOMAoptions{fontsize=9pt}
    	\large\@streszczenieang
	}
}{
	\ifthenelse{\streszczenieangdlugosc > \dlugoscmaxa}{
		\parbox{\textwidth}{
		\@streszczenieang
		}
	}{
		\parbox{\textwidth}{
		\footnotesize
		\@streszczenieang
		}
	}
}
\end{flushleft}
\end{minipage}

\vspace{\fill}
\begin{minipage}[b][0.33\textheight]{\textwidth}
\flushright
\begin{tabular}{l}
 Supervisor:\\
	{\@opiekun}
\end{tabular}

\vspace{0.8ex}
\emph{
\footnotesize
\begin{flushleft}
For the purposes of archival thesis qualified to:*
\vspace{-1ex}
\begin{enumerate}
	\itemsep0em 
    \item[a)] category A (perpetual files)
    \item[b)] category BE 50 (subject to expertise after 50 years)
\end{enumerate}
\vspace{-1ex}
* delete as appropriate 
\end{flushleft}
}
\begin{flushright}
\boxed{~~~\textrm{stamp of the faculty}~~~}
\end{flushright}
\vspace{0cm}
\begin{center}
\large Wrocław, \the\year
\end{center}
\end{minipage}
%\end{flushleft}
\clearpage
\restoregeometry
}
}

\newif\if@english
\newif\if@magisterska
\newif\if@licencjacka
\newif\if@inzynierska

\DeclareOption{english}{\@englishtrue}
\DeclareOption{polski}{\@englishfalse}
\DeclareOption{magisterska}{\@magisterskatrue}
\DeclareOption{licencjacka}{\@licencjackatrue}
\DeclareOption{inzynierska}{\@inzynierskatrue}

\ExecuteOptions{polski}
\ExecuteOptions{magisterska}
\ProcessOptions\relax

\if@english
	\RequirePackage[main=english, polish]{babel}
	\renewcommand{\maketitle}{
		\pagenumbering{gobble}
        \stronatytulowaang
		\cleardoublepage
	}
\else
	\RequirePackage[main=polish,english]{babel}
	\AtBeginDocument{
  		\renewcommand\tablename{Tabela}
	}    
    \renewcommand{\maketitle}{
		\pagenumbering{gobble}
        \stronatytulowapl
		\cleardoublepage
        \stronatytulowaang
		\cleardoublepage
	}
\fi

\if@magisterska
	\rodzajpracy{magisterska}
	\rodzajpracyang{Master's Thesis}
\fi
\if@licencjacka
	\rodzajpracy{licencjacka}
	\rodzajpracyang{Bachelor's Thesis}
\fi
\if@inzynierska
	\rodzajpracy{inżynierska}
	\rodzajpracyang{Engineering Thesis}
\fi



\RequirePackage{fancyhdr}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\renewcommand{\sectionmark}[1]{\markright{#1}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[LO]{\itshape\nouppercase{\rightmark}}
\fancyhead[RE]{{\leftmark}}
\renewcommand{\headrulewidth}{0pt}

\newcommand{\bibliografia}[1]{
	\phantomsection
	\if@english
		\addcontentsline{toc}{chapter}{Bibliography}
	\else
		\addcontentsline{toc}{chapter}{Bibliografia}
	\fi
	\bibliography{#1}
}