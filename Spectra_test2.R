library(prospectr)
data <- read.csv("spectra/E (4).TXT", header=F)
wn_range <- c(300, 1600)

data <- subset(data, V1 >= wn_range[1] & V1 <= wn_range[2])
spectra <- as.matrix(t(data[-1]))
colnames(spectra) <- data[[1]]

smoothed_spectra <- savitzkyGolay(X = spectra,
                                  m = 0,
                                  p = 0,
                                  w = 3)



matplot(x = as.numeric(colnames(spectra)),
        y = t(spectra),
        lty = 1, 
        type = "l",
        col="red",
        xlab = "Wavenumber [1/cm]",
        ylab = "Intensity [-]",
        xlim = c(1600, 300),
        font = 2, 
        font.lab = 2,  ###font = 2 ist Fett gedruckt
        lab = c(20,15,10), 
        bty = "l", 
        family = "sans", 
        xaxs = "i")

matlines(x = as.numeric(colnames(smoothed_spectra)),
         y = t(smoothed_spectra),
         lty = 1,
         col = "blue")
