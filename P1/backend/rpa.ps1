

# ================= CONFIGURACIÓN =================
$txtPath = "C:\Users\Douglas\Desktop\mails.txt"

$smtpServer = "smtp.gmail.com"
$smtpPort   = 587
$fromEmail  = "dougdari@gmail.com"
$password   = ""
# ================================================

$securePass = ConvertTo-SecureString $password -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential ($fromEmail, $securePass)

$emails = Get-Content $txtPath

foreach ($to in $emails) {
    Write-Host "Enviando correo a $to"

    Send-MailMessage `
        -From $fromEmail `
        -To $to `
        -Subject "Correo automático RPA" `
        -Body "Este correo fue enviado por un RPA ejecutado en Windows." `
        -SmtpServer $smtpServer `
        -Port $smtpPort `
        -UseSsl `
        -Credential $cred

    Start-Sleep -Seconds 1
}
