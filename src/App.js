import React from 'react';
import Main from './components/fileUpload';
import { useState, useEffect } from 'react'
import axios from 'axios'



const App = () => {
  useEffect(() => {
    console.log('effect')
    axios
      .get('http://localhost:5000/')
      .then(response => {
        console.log('promise fulfilled')
        console.log(response)
      })
  }, [])

  return(
    <div>
      <h1>File Upload</h1>
      <Main />
    </div>
  );
}




export default App;