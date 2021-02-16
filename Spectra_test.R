library("baseline")
library("prospectr")

data <- read.csv("spectra/E (4).TXT", header=F)
wn_range <- c(300, 1600)
threshold <- 3.5E-5


data <- subset(data, V1 >= wn_range[1] & V1 <= wn_range[2])


spectra <- as.matrix(t(data[-1]))
colnames(spectra) <- data[[1]]

mean_intensity <- mean(spectra[,1])
threshold_pd <- mean_intensity * threshold


pd <- baseline.peakDetection(as.matrix(spectra),
                       left=1,
                       right=1,
                       lwin=20,
                       rwin=20,
                       snminimum=2)


diffspectra2 <- savitzkyGolay(X = spectra,
                              m = 2,
                              p = 3,
                              w = 25)


diffspectra3 <- savitzkyGolay(X = spectra,
                              m = 3,
                              p = 3,
                              w = 25)


zeros <- t(diff(t(sign(diffspectra3))))

peaks <- (zeros > 0) & (diffspectra2[-1] < -threshold_pd)

peaks_wns <- as.numeric(colnames(peaks)[which(peaks)])

peak_condensing <- c()
for (i in 1:ncol(diffspectra2)){
        if (is.element(as.numeric(colnames(diffspectra2))[i],peaks_wns)){
                peak_condensing <- c(peak_condensing, 
                                     as.numeric(colnames(diffspectra2)[i]))
        } else if (diffspectra2[1,i] > 0){
                if (length(peak_condensing) > 1){
                        peaks_wns <- setdiff(peaks_wns, peak_condensing)
                        peaks_wns <- c(peaks_wns, mean(peak_condensing))
                }
                peak_condensing <- c()
        }
}
peaks_wns <- setdiff(peaks_wns, peak_condensing)
peaks_wns <- c(peaks_wns, mean(peak_condensing))




peaks_wns <- sort(peaks_wns)


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

matplot(x = as.numeric(colnames(diffspectra2)),
        y = t(diffspectra2),
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

matplot(x = as.numeric(colnames(diffspectra3)),
        y = t(diffspectra3),
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


grid(lwd = 0.8)
abline(h=0)
abline(v=peaks_wns)


print(peaks_wns)

