library("baseline")
library("ChemoSpec")

data <- read.csv("spectra/E (2).TXT", header=F)
wn_range <- c(300, 1600)
data <- subset(data, V1 >= wn_range[1] & V1 <= wn_range[2])


spectra <- as.data.frame(t(data[-1]))
colnames(spectra) <- data[[1]]

pd <- baseline.peakDetection(as.matrix(spectra),
                       left=5,
                       right=5,
                       lwin=5,
                       rwin=5,
                       snminimum=2)

wavenumbers <- as.numeric(colnames(spectra))


matplot(x = colnames(spectra),
        y = t(as.matrix(spectra)),
        lty = 1, 
        type = "l",
        col="red",
        xlab = "Wavenumber [1/cm]",
        ylab = "Intensity [-]",
        font = 2, 
        font.lab = 2,  ###font = 2 ist Fett gedruckt
        lab = c(20,15,10), 
        bty = "l", 
        family = "sans", 
        xaxs = "i")
grid(lwd = 0.8)

peak_wns <- wavenumbers[pd$peaks[[1]]]
abline(v=peak_wns)
