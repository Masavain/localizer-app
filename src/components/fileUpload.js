import React from 'react';
import axios from 'axios'

class Main extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      imageURL: '',
      plotURL: '',
    };

    this.handleUploadImage = this.handleUploadImage.bind(this);
  }

  handleUploadImage(e) {
    e.preventDefault();
    let file = this.state.fileToBeSent;

    const data = new FormData();
    data.append('file', this.uploadInput.files[0]);
    data.append('filename', this.fileName.value);
    console.log(data)
    axios
      .post("http://localhost:5000/upload", data)
      .then(res => {
        console.log('response')
        console.log(res.data)
        console.log('---')
        this.setState({imageURL : `http://localhost:5000/img/${res.data}`,
                          plotURL: `http://localhost:5000/plot/plot.png`})})
      .catch(err => console.warn(err));

    // ev.preventDefault();

    // const data = new FormData();
    // data.append('file', this.uploadInput.files[0]);
    // data.append('filename', this.fileName.value);

    // fetch('http://localhost:5000/upload', {
    //   method: 'POST',
    //   body: data,
    // }).then((response) => {
    //   response.json().then((body) => {
    //     this.setState({ imageURL: `http://localhost:5000/${body.file}` });
    //   });
    // });
  }

  render() {
    return (
      <form onSubmit={this.handleUploadImage}>
        <div>
          <input ref={(ref) => { this.uploadInput = ref; }} type="file" />
        </div>
        <div>
          <input ref={(ref) => { this.fileName = ref; }} type="text" placeholder="Enter the desired name of file" />
        </div>
        <br />
        <div>
          <button>Upload</button>
        </div>
        <img src={this.state.imageURL} alt="img" />
        <img src={this.state.plotURL} alt="plt" />
      </form>
    );
  }
}

export default Main;