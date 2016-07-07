


var express = require('express');
var path = require('path');
var fs = require('fs');

var app = express();
var PORT = 10111;


app.set('port', PORT);
app.use(express.static(path.join(__dirname, 'public')));



// endpoint setup ------------------------------------------------------

app.get('/', function (req, res, next) {
  var index_html = fs.readFileSync("./public/html/tsne_svg.html", "utf-8")
  res.send(index_html)
})


app.post('/hello', function(req, res, next) {



})

// endpoint setup end ------------------------------------------------------



app.listen(app.get('port'), function() {
  console.log('Express server listening on port ' + app.get('port'));
});






