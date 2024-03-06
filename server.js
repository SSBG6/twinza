const express = require('express');
const app = express();
const port = process.env.PORT || 4000;
app.use(express.urlencoded({extended:true}));

// Set the view engine to 'ejs'
app.set('view engine', 'ejs');

// importing files
app.use(express.static('public/css'));//css
app.use(express.static('public/imgs'));//images

// Set index.html as the home page
app.get("/", (req, res) => {
    res.render("index"); // Express will automatically look for "index.ejs" in the "views" directory
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
