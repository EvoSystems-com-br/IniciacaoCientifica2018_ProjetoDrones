
var droneId = 'drone1'
var secret = 'dr0neRulez2A5T7U'

var mandarButton = document.getElementById("mandarButton");
var myCommand = "empty";
var statusAtual = "Iniciando";

function sendCommand(){
  var xmlhttp = new XMLHttpRequest();
  var statusButton = document.getElementById("statusButton");
  if(statusButton.value=="Preparar"){
    var url = "https://3vkeycenej.execute-api.us-east-1.amazonaws.com/prod/CIAB-2018-DroneCommandQueue?operation=PUSH&secret=" + secret + "&droneId=" + droneId + "&command=" + "preparar";
  }else{
    var url = "https://3vkeycenej.execute-api.us-east-1.amazonaws.com/prod/CIAB-2018-DroneCommandQueue?operation=PUSH&secret=" + secret + "&droneId=" + droneId + "&command=" + myCommand;
  }
  xmlhttp.onreadystatechange = function() {
    if (xmlhttp.readyState == 4) {
      var jsonObject = null;
    }
  };
  xmlhttp.open("GET", url, true);
  xmlhttp.send();
}

function commandPressed(command){
  myCommand = command;
}

function videoPressed(command){
  var video = document.getElementById("myVideo");
  switch (command) {
    case 'demo':
      video.src = "https://www.youtube.com/embed/LIVAmrgerTI?controls=1&autoplay=1&rel=0&showinfo=0&modestbranding=0";
      break;
    case 'espiral':
      video.src = "https://www.youtube.com/embed/YTtv2LQ6-nk?autoplay=1&controls=1&rel=0&showinfo=0&modestbranding=0";
      break;
    case 'esquadrilha':
      video.src = "https://www.youtube.com/embed/VOVLKRiJrVg?autoplay=1&controls=1&rel=0&showinfo=0&modestbranding=0";
      break;
    default:
      video.src = "https://www.youtube.com/embed/LIVAmrgerTI?autoplay=0&controls=0&rel=0&showinfo=0&modestbranding=0";
      break;

  }

}


function getStatus(){
  var xmlhttp = new XMLHttpRequest();
  var urlPeekStatus = "https://3vkeycenej.execute-api.us-east-1.amazonaws.com/prod/CIAB-2018-DroneCommandQueue?droneId=status&operation=PEEK&secret=dr0neRulez2A5T7U"
  xmlhttp.onreadystatechange = function() {
    if (this.readyState == 4 ) {
      var statusButton = document.getElementById("statusButton");
      var status = xmlhttp.responseText;
      document.getElementById('statusText').innerHTML= status;

      if(status != statusAtual){
        if(status =='Aguardando'){
          statusButton.value = "Preparar";
          statusButton.style.fontSize = "32px";
          statusButton.style.color = "#ffffff";
          statusButton.style.backgroundColor = "#a09d9d";
          statusButton.addEventListener("focus", function () {
            statusButton.style.color = "#ffffff";
            statusButton.style.backgroundColor = "#a09d9d";
          });
          statusButton.addEventListener("blur", function () {
            statusButton.style.color = "#ffffff";
            statusButton.style.backgroundColor = "#a09d9d";
          });
        }else if(status=='Disponível'){
          statusButton.value = "Preparar";
          statusButton.style.fontSize = "32px";
          statusButton.style.color = "#BC3F3F";
          statusButton.style.backgroundColor = "#F4D9D9";
          statusButton.addEventListener("focus", function () {
            statusButton.style.color = "#F4D9D9";
            statusButton.style.backgroundColor = "#932F2F";
          });
          statusButton.addEventListener("blur", function () {
            statusButton.style.color = "#BC3F3F";
            statusButton.style.backgroundColor = "#F4D9D9";
          });
         }else if(status=='Pronto! Aguardando comando') {
           statusButton.value = "Enviar";
           statusButton.style.fontSize = "40px";
           statusButton.style.color = "#BC3F3F";
           statusButton.style.backgroundColor = "#F4D9D9";
           statusButton.addEventListener("focus", function () {
             statusButton.style.color = "#F4D9D9";
             statusButton.style.backgroundColor = "#932F2F";
           });
           statusButton.addEventListener("blur", function () {
             statusButton.style.color = "#BC3F3F";
             statusButton.style.backgroundColor = "#F4D9D9";
           });
         } else{
           statusButton.value = "Enviar";
           statusButton.style.fontSize = "40px";
           statusButton.style.color = "#ffffff";
           statusButton.style.backgroundColor = "#a09d9d";
           statusButton.addEventListener("focus", function () {
             statusButton.style.color = "#ffffff";
             statusButton.style.backgroundColor = "#a09d9d";
           });
           statusButton.addEventListener("blur", function () {
             statusButton.style.color = "#ffffff";
             statusButton.style.backgroundColor = "#a09d9d";
           });
        }

        statusAtual=status;
      }
    }
  };

  xmlhttp.open("GET", urlPeekStatus, true);
  xmlhttp.send();

  setTimeout(getStatus, 1000);

}
function setVideoRate(){
   var video = document.getElementById("myVideo");
  var width = video.clientHeight*(16/9);
  video.style.width = width;
}

getStatus()
setVideoRate()
