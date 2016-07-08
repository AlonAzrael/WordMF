


var express = require('express');
var path = require('path');
var fs = require('fs');
var logger = require('morgan')

var app = express();
var PORT = 10111;


app.set('port', PORT);
app.use(logger('dev'))
app.use(express.static(path.join(__dirname, 'public')));

function print (argument) {
    console.log.apply(console, arguments)
}


// endpoint setup ------------------------------------------------------

app.get('/', function (req, res, next) {
  var index_html = fs.readFileSync("./public/html/tsne_svg.html", "utf-8")
  res.send(index_html)
})


app.post('/get_tsne_coor_data', function(req, res, next) {

    var word_tsne_coor_txt = fs.readFileSync("../__X_data/word_tsne_coor.txt", "utf-8")
    res.charset = 'utf-8';
    res.send(word_tsne_coor_txt)

})

// endpoint setup end ------------------------------------------------------



app.listen(app.get('port'), function() {
  console.log('Express server listening on port ' + app.get('port'));
});






